"""Reinforcement learning agent for quantum noise modeling."""

from pathlib import Path
from typing import Optional, Union, Dict, Any
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
        policy_kwargs = dict(
            features_extractor_class=CNNFeaturesExtractor,
            features_extractor_kwargs=dict(
                features_dim=agent_config.features_dim,
                filter_shape=(env.n_qubits, agent_config.filter_size),
                n_filters=agent_config.n_filters,
            ),
            net_arch=dict(
                pi=agent_config.pi_net_arch,
                vf=agent_config.vf_net_arch
            )
        )
        
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
    
    def train(
        self,
        total_timesteps: int,
        check_freq: int = 1000,
        save_path: Optional[str] = None,
        save_best: bool = True,
        progress_bar: bool = True,
    ) -> Dict[str, Any]:
        """Train the agent.
        
        Args:
            total_timesteps: Total training timesteps
            check_freq: Steps between evaluations
            save_path: Path to save best model
            save_best: Whether to save best model
            progress_bar: Whether to show progress bar
        
        Returns:
            Dictionary with training results
        """
        # Create callback
        callback = TrainingCallback(
            env=self.env,
            check_freq=check_freq,
            save_path=save_path,
            save_best=save_best,
            verbose=1,
        )
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            progress_bar=progress_bar,
            callback=callback,
        )
        
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
            obs, _, terminated, _, _ = temp_env.step(action)
        
        # Return result
        if return_qibo:
            return self.env.encoder.array_to_circuit(temp_env.current_circuit)
        else:
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
                obs, reward, done, truncated, info = self.env.step(action)
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
