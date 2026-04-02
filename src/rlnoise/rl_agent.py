"""Reinforcement learning agent for quantum noise modeling."""

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from stable_baselines3 import PPO

from rlnoise.gym_env import QuantumCircuitEnv
from rlnoise.neural_network import CNNFeaturesExtractor
from rlnoise.callback import TrainingCallback
from rlnoise.config import AgentConfig


class RLAgent:
    """Reinforcement learning agent using PPO for quantum noise modeling.

    The agent learns to apply appropriate noise parameters to quantum circuits
    to match target noisy behavior. Uses a CNN feature extractor to process
    the sliding window observations.

    Args:
        env: QuantumCircuitEnv for training
        agent_config: AgentConfig with hyperparameters
        model_path: Optional path to load pre-trained model

    Example:
        >>> from rlnoise.config import AgentConfig
        >>> config = AgentConfig(
        ...     features_dim=64,
        ...     filter_size=3,
        ...     n_filters=32,
        ...     n_steps=2048,
        ...     batch_size=64
        ... )
        >>> agent = RLAgent(env, config)
        >>> agent.train(total_timesteps=100000, check_freq=1000)
    """

    def __init__(
        self,
        env: QuantumCircuitEnv,
        agent_config: AgentConfig,
        model_path: Optional[str] = None,
    ):
        """Initialize the RL agent.

        Args:
            env: Training environment
            agent_config: Agent configuration
            model_path: Path to pre-trained model to load
        """
        self.env = env
        self.agent_config = agent_config

        # Build policy kwargs with CNN feature extractor
        policy_kwargs = {
            "features_extractor_class": CNNFeaturesExtractor,
            "features_extractor_kwargs": {
                "features_dim": agent_config.features_dim,
                "filter_shape": (env.n_qubits, agent_config.filter_size),
                "n_filters": agent_config.n_filters,
            },
            "net_arch": {
                "pi": agent_config.pi_net_arch,
                "vf": agent_config.vf_net_arch,
            },
        }

        # Load or create model
        if model_path is not None:
            self.load(model_path)
        else:
            self.model = PPO(
                agent_config.policy,
                env,
                policy_kwargs=policy_kwargs,
                n_steps=agent_config.n_steps,
                batch_size=agent_config.batch_size,
                learning_rate=agent_config.learning_rate,
                gamma=agent_config.gamma,
                clip_range=agent_config.clip_range,
                verbose=agent_config.verbose,
            )

    def __str__(self) -> str:
        """Summary showing input/action dimensions and parameter counts."""
        obs_shape = self.model.observation_space.shape
        action_space = self.model.action_space
        action_shape = action_space.shape
        action_low = (  # type: ignore[union-attr]
            float(action_space.low.flat[0]) if hasattr(action_space, "low") else None
        )
        action_high = (  # type: ignore[union-attr]
            float(action_space.high.flat[0]) if hasattr(action_space, "high") else None
        )

        total_params = sum(p.numel() for p in self.model.policy.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.policy.parameters() if p.requires_grad
        )

        lines = [
            "=" * 60,
            "RLAgent (PPO)",
            "=" * 60,
            f"  Observation space:    {obs_shape}",
            f"  Action space:         {action_shape}",
            f"  Action range:         [{action_low:.3f}, {action_high:.3f}]"
            if action_low is not None
            else "  Action range:         N/A",
            "-" * 60,
            f"  Total parameters:     {total_params:,}",
            f"  Trainable parameters: {trainable_params:,}",
            "-" * 60,
            f"  Features dim:         {self.agent_config.features_dim}",
            f"  CNN filters:          {self.agent_config.n_filters}",
            f"  CNN filter size:      {self.agent_config.filter_size}",
            f"  Policy net arch:      {self.agent_config.pi_net_arch}",
            f"  Value net arch:       {self.agent_config.vf_net_arch}",
            "-" * 60,
            f"  Learning rate:        {self.agent_config.learning_rate}",
            f"  Batch size:           {self.agent_config.batch_size}",
            f"  N steps:              {self.agent_config.n_steps}",
            f"  Gamma:                {self.agent_config.gamma}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def print_network(self) -> None:
        """Print the neural network architecture.

        Shows three levels of detail:
        1. PyTorch's built-in module tree (layers and their shapes).
        2. Parameter count per named sub-module.
        3. All parameter tensor shapes with trainability flag.
        """
        policy = self.model.policy

        print("=" * 70)
        print("NEURAL NETWORK ARCHITECTURE")
        print("=" * 70)
        print()
        print(policy)

        print()
        print("=" * 70)
        print("PARAMETER COUNT BY MODULE")
        print("=" * 70)
        total = 0
        for name, module in policy.named_modules():
            params = sum(p.numel() for p in module.parameters(recurse=False))
            if params > 0:
                trainable = sum(
                    p.numel()
                    for p in module.parameters(recurse=False)
                    if p.requires_grad
                )
                print(
                    f"  {name:50s}  {params:>8,} params"
                    f"  (trainable: {trainable:>8,})"
                )
                total += params
        print("-" * 70)
        print(f"  {'TOTAL':50s}  {total:>8,} params")
        print("=" * 70)

        print()
        print("=" * 70)
        print("PARAMETER TENSOR SHAPES")
        print("=" * 70)
        for name, param in policy.named_parameters():
            grad_flag = "grad" if param.requires_grad else "no-grad"
            shape_str = str(list(param.shape))
            print(
                f"  [{grad_flag:7s}]  {name:55s}"
                f"  {shape_str:>25s}  ({param.numel():,})"
            )
        print("=" * 70)

    def train(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        total_timesteps: int,
        check_freq: int = 1000,
        save_path: Optional[str] = None,
        save_best: bool = True,
        progress_bar: bool = True,
        verbose: bool = True,
        deterministic_train_eval: bool = True,
        history_path: Optional[str] = None,
        previous_history: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Train the agent.

        Args:
            total_timesteps: Total training timesteps
            check_freq: Steps between evaluations
            save_path: Path to save best model
            save_best: Whether to save best model
            progress_bar: Whether to show progress bar
            verbose: If True, print average reward at each evaluation step
            deterministic_train_eval: If True (default), evaluate training
                performance with a full deterministic pass over all training
                circuits at each check step.  If False, use rewards accumulated
                from the live rollout (faster but noisier).
            history_path: Optional path to save training history as a .npz
                file (e.g. ``"results/run1_history"``).  The ``.npz`` extension
                is added automatically.  The file can be reloaded later with
                :meth:`~rlnoise.callback.TrainingCallback.load_history` and
                passed to :func:`~rlnoise.visualization.plot_training_dashboard`.
            previous_history: Optional history dict returned by a previous
                :meth:`train` call (or loaded with
                :meth:`~rlnoise.callback.TrainingCallback.load_history`).
                When provided, the new callback is pre-populated with all
                prior timestep/result data so that the returned history and
                any saved ``.npz`` file contain the full combined curve.

        Returns:
            Dictionary with training results
        """
        # Capture stdout NOW, before rich's progress bar wraps sys.stdout.
        # The callback will write to this stream so output reaches the notebook.
        out_stream = sys.stdout

        # When save_best=True but no explicit path, save to a temp directory so
        # we can restore the best weights at the end of training.
        _temp_dir: Optional[str] = None
        effective_save_path = save_path
        if save_best and save_path is None:
            _temp_dir = tempfile.mkdtemp()
            effective_save_path = str(Path(_temp_dir) / "best_model")

        # Create callback
        callback = TrainingCallback(
            env=self.env,
            check_freq=check_freq,
            save_path=effective_save_path,
            save_best=save_best,
            verbose=int(verbose),
            out_stream=out_stream,
            deterministic_train_eval=deterministic_train_eval,
            verbose_save=save_path is not None,  # suppress print for temp path
        )

        # Pre-populate the callback with data from a previous training run so
        # that the returned history (and any saved .npz) spans both runs.
        if previous_history is not None:
            callback.timestep_list = list(previous_history["timesteps"])
            callback.train_results = [np.array(r) for r in previous_history["train_results"]]
            callback.eval_results = [np.array(r) for r in previous_history["eval_results"]]
            callback.best_mean_reward = float(previous_history["best_mean_reward"])

        # Suppress PPO's own tabular output; the callback handles all printing
        original_verbose = self.model.verbose
        self.model.verbose = 0
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                progress_bar=progress_bar,
                callback=callback,
            )
        finally:
            self.model.verbose = original_verbose

        # Restore best model weights if save_best was used and a best model was found
        if save_best and effective_save_path is not None and callback.best_mean_reward > -np.inf:
            loaded = PPO.load(effective_save_path)
            self.model.policy.load_state_dict(loaded.policy.state_dict())
            if save_path is not None:
                print(f"Loaded best model weights from {save_path}", file=out_stream, flush=True)
            else:
                print("Loaded best model weights.", file=out_stream, flush=True)

        # Clean up temp directory
        if _temp_dir is not None:
            shutil.rmtree(_temp_dir, ignore_errors=True)

        # Optionally persist the training history
        if history_path is not None:
            callback.save_history(history_path)
            print(f"Training history saved to {history_path}.npz", file=out_stream, flush=True)

        # Return training results
        return callback.get_results()

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Predict action for given observation.

        Args:
            observation: Circuit observation
            deterministic: Whether to use deterministic policy

        Returns:
            Action array of noise parameters
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def apply_to_circuit(
        self,
        circuit_array: np.ndarray,
        return_qibo: bool = True,
    ):
        """Apply learned noise policy to a circuit.

        Args:
            circuit_array: Circuit array representation
            return_qibo: If True, return Qibo circuit, else array

        Returns:
            Noisy circuit (Qibo circuit or array)
        """
        # Create temporary environment with single circuit
        # pylint: disable=import-outside-toplevel
        from rlnoise.dataset import CircuitDataset

        # Expand dims if needed to get (n_circuits, n_moments, n_qubits, encoding_dim)
        if circuit_array.ndim == 2:
            circuit_array = np.expand_dims(circuit_array, axis=0)
        if circuit_array.ndim == 3:
            circuit_array = np.expand_dims(circuit_array, axis=0)

        # Get number of qubits from circuit shape
        n_qubits = circuit_array.shape[2]

        # Create dummy labels (won't be used for inference)
        dummy_labels = np.zeros((1, 2**n_qubits, 2**n_qubits), dtype=complex)

        # Create single-circuit dataset
        temp_dataset = CircuitDataset(
            circuits=circuit_array,
            labels=dummy_labels,
        )

        # Create temporary environment
        temp_env = QuantumCircuitEnv(
            dataset=temp_dataset,
            encoder=self.env.encoder,
            env_config=self.env.env_config,
            reward_config=self.env.reward_config,
        )

        # Run episode
        obs, _ = temp_env.reset(options={"circuit_idx": 0})
        terminated = False

        while not terminated:
            action = self.predict(obs, deterministic=True)
            obs, _, terminated, _, _ = temp_env.step(action)  # noqa

        # Return result
        if return_qibo:
            return self.env.encoder.array_to_circuit(temp_env.current_circuit)
        return temp_env.current_circuit

    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """Evaluate agent performance.

        Args:
            n_episodes: Number of episodes to evaluate
            deterministic: Use deterministic policy

        Returns:
            Dictionary with evaluation metrics
        """
        rewards = []

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.predict(obs, deterministic=deterministic)
                obs, reward, done, _, _ = self.env.step(action)
                episode_reward = reward  # Only final reward matters

            rewards.append(episode_reward)

        rewards = np.array(rewards)
        return {
            "mean_reward": rewards.mean(),
            "std_reward": rewards.std(),
            "min_reward": rewards.min(),
            "max_reward": rewards.max(),
        }

    def save(self, path: str):
        """Save the model.

        Args:
            path: Path to save model (without extension)
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    def load(self, path: str):
        """Load a pre-trained model.

        Args:
            path: Path to model file
        """
        self.model = PPO.load(path, env=self.env)
