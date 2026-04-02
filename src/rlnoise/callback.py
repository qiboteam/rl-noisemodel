"""Training callbacks for monitoring and evaluation."""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from rlnoise.gym_env import QuantumCircuitEnv


class TrainingCallback(BaseCallback):  # pylint: disable=too-many-instance-attributes
    """Callback for monitoring training progress and evaluation.

    Periodically evaluates the model on training and validation sets,
    tracks metrics like reward and fidelity, and optionally saves the best model.

    Args:
        env: QuantumCircuitEnv to evaluate on
        check_freq: Evaluate every check_freq steps
        save_path: Optional path to save best model
        save_best: Whether to save best model based on validation reward
        verbose: Verbosity level (0=none, 1=info, 2=debug)

    Example:
        >>> callback = TrainingCallback(
        ...     env=env,
        ...     check_freq=1000,
        ...     save_path="models/best_model",
        ...     save_best=True,
        ...     verbose=1
        ... )
        >>> model.learn(total_timesteps=10000, callback=callback)
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-positional-arguments
        self,
        env: QuantumCircuitEnv,
        check_freq: int = 1000,
        save_path: Optional[str] = None,
        save_best: bool = True,
        verbose: int = 1,
        out_stream=None,
        deterministic_train_eval: bool = True,
        verbose_save: bool = True,
    ):
        """Initialize the training callback.

        Args:
            env: Environment for evaluation
            check_freq: Steps between evaluations
            save_path: Path to save best model (without extension)
            save_best: Whether to save best model
            verbose: Verbosity level
            out_stream: Output stream for prints (captured before rich wraps stdout)
            deterministic_train_eval: If True (default), evaluate training performance
                with a full deterministic pass over all training circuits at each
                check step.  If False, accumulate terminal rewards/metrics from the
                live rollout between check steps instead (faster but noisier).
            verbose_save: If False, suppress the "Model saved in ..." print (used
                when saving to a temporary path).
        """
        super().__init__(verbose)

        self.env = env
        self.check_freq = check_freq
        self.save_path = save_path
        self.save_best = save_best
        self._verbose_save = verbose_save
        self._out = out_stream if out_stream is not None else sys.stdout
        self.has_val_set = env.n_circuits > env.n_circuits_train
        self.deterministic_train_eval = deterministic_train_eval

        # Initialize tracking
        self.best_mean_reward = -np.inf
        self.eval_results = []
        self.train_results = []
        self.timestep_list = []
        self._metric_name = env.reward_config.metric

        # Rollout accumulators (used only when deterministic_train_eval=False)
        self._rollout_rewards: list = []
        self._rollout_metric_values: list = []

        if not self.has_val_set:
            print(
                "Warning: no validation set found (val_split=0.0). "
                "Evaluation rewards will be recorded as 0.0.",
                file=self._out,
            )

        # Create save directory if needed
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        """Called after each step during training."""
        if not self.deterministic_train_eval:
            # Accumulate terminal rewards/metrics from the live rollout
            dones = self.locals.get("dones", [])
            rewards = self.locals.get("rewards", [])
            infos = self.locals.get("infos", [])
            for done, reward, info in zip(dones, rewards, infos):
                if done:
                    self._rollout_rewards.append(float(reward))
                    if info is not None and "metric" in info:
                        self._rollout_metric_values.append(float(info["metric"]))

        if self.n_calls % self.check_freq == 0:
            self._evaluate()
        return True

    def _evaluate(self):
        """Evaluate the current policy on train and validation sets."""
        if self.deterministic_train_eval:
            train_metrics = self._evaluate_on_set(train=True)
        else:
            # Use rewards/metrics accumulated from the live rollout
            if self._rollout_rewards:
                r = np.array(self._rollout_rewards)
                m = (
                    np.array(self._rollout_metric_values)
                    if self._rollout_metric_values else np.zeros(0)
                )
                train_metrics = np.array([
                    r.mean(), r.std(),
                    m.mean() if len(m) else 0.0,
                    m.std() if len(m) else 0.0,
                ])
            else:
                train_metrics = np.zeros(4)
            self._rollout_rewards = []
            self._rollout_metric_values = []
        self.train_results.append(train_metrics)

        # Evaluate on validation set, or store zeros if none exists
        if self.has_val_set:
            val_metrics = self._evaluate_on_set(train=False)
        else:
            val_metrics = np.zeros(4)
        self.eval_results.append(val_metrics)

        # Store timestep
        self.timestep_list.append(self.num_timesteps)

        # Print single-line summary
        if self.verbose > 0:
            msg = (
                f"Step {self.num_timesteps:>7d} | "
                f"Train reward: {train_metrics[0]:.4f} Â± {train_metrics[1]:.4f}  "
                f"{self._metric_name}: {train_metrics[2]:.4f} Â± {train_metrics[3]:.4f}  |  "
                f"Val reward: {val_metrics[0]:.4f} Â± {val_metrics[1]:.4f}  "
                f"{self._metric_name}: {val_metrics[2]:.4f} Â± {val_metrics[3]:.4f}"
            )
            print(msg, file=self._out, flush=True)

        # Save best model based on validation reward (or training if no val set)
        if self.save_best and self.save_path is not None:
            mean_reward = val_metrics[0] if self.has_val_set else train_metrics[0]

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.save_path)
                if self._verbose_save:
                    print(f"Model saved in {self.save_path}", file=self._out, flush=True)

    def _evaluate_on_set(self, train: bool = True) -> np.ndarray:
        """Run deterministic episodes over the validation set.

        Returns:
            Array [mean_reward, std_reward, mean_metric, std_metric]
        """
        start = 0 if train else self.env.n_circuits_train
        stop = self.env.n_circuits_train if train else self.env.n_circuits

        rewards = []
        metric_values = []

        for i in range(start, stop):
            # Reset to specific circuit
            obs, _ = self.env.reset(options={"circuit_idx": i})
            done = False
            reward = 0.0
            info: dict = {}

            while not done:
                # Get deterministic action from model
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = self.env.step(action)

            rewards.append(reward)
            if "metric" in info:
                metric_values.append(info["metric"])

        rewards = np.array(rewards)
        if metric_values:
            metric_values = np.array(metric_values)
            return np.array([
                rewards.mean(), rewards.std(),
                metric_values.mean(), metric_values.std(),
            ])
        return np.array([rewards.mean(), rewards.std(), 0.0, 0.0])

    def _print_metrics(self, set_name: str, metrics: np.ndarray):
        """Print evaluation metrics."""
        print(
            f"{set_name}: reward {metrics[0]:.4f} \u00b1 {metrics[1]:.4f}",
            file=self._out, flush=True
        )

    def get_results(self) -> dict:
        """Get all recorded results.

        Returns:
            Dictionary with timesteps, results for train/val sets, and metric name.
            Each entry in train_results/eval_results is
            [mean_reward, std_reward, mean_metric, std_metric].
        """
        return {
            "timesteps": self.timestep_list,
            "train_results": self.train_results,
            "eval_results": self.eval_results,
            "best_mean_reward": self.best_mean_reward,
            "metric_name": self._metric_name,
            "check_freq": self.check_freq,
            "n_qubits": self.env.n_qubits,
            "n_circuits_train": self.env.n_circuits_train,
            "n_circuits_val": self.env.n_circuits - self.env.n_circuits_train,
        }

    def save_history(self, filepath: str) -> None:
        """Save the full training history to a .npz file.

        The saved file can be reloaded with :meth:`load_history` and passed
        directly to :func:`~rlnoise.visualization.plot_training_dashboard`.

        Args:
            filepath: Destination path.  The ``.npz`` extension is added
                automatically if omitted.
        """
        if not filepath.endswith(".npz"):
            filepath = filepath + ".npz"
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        results = self.get_results()
        np.savez(
            filepath,
            timesteps=np.array(results["timesteps"]),
            train_results=np.array(results["train_results"]),
            eval_results=np.array(results["eval_results"]),
            best_mean_reward=np.array(results["best_mean_reward"]),
            metric_name=np.array(results["metric_name"]),
            check_freq=np.array(results["check_freq"]),
            n_qubits=np.array(results["n_qubits"]),
            n_circuits_train=np.array(results["n_circuits_train"]),
            n_circuits_val=np.array(results["n_circuits_val"]),
        )

    @staticmethod
    def load_history(filepath: str) -> dict:
        """Load training history previously saved with :meth:`save_history`.

        The returned dictionary is compatible with
        :func:`~rlnoise.visualization.plot_training_dashboard`.

        Args:
            filepath: Path to the ``.npz`` file.  The extension is added
                automatically if omitted.

        Returns:
            Dictionary with keys ``timesteps``, ``train_results``,
            ``eval_results``, ``best_mean_reward``, ``metric_name``,
            ``check_freq``, ``n_qubits``, ``n_circuits_train``,
            ``n_circuits_val``.
        """
        if not filepath.endswith(".npz"):
            filepath = filepath + ".npz"
        data = np.load(filepath, allow_pickle=True)
        return {
            "timesteps": data["timesteps"].tolist(),
            "train_results": data["train_results"].tolist(),
            "eval_results": data["eval_results"].tolist(),
            "best_mean_reward": float(data["best_mean_reward"]),
            "metric_name": str(data["metric_name"]),
            "check_freq": int(data["check_freq"]),
            "n_qubits": int(data["n_qubits"]),
            "n_circuits_train": int(data["n_circuits_train"]),
            "n_circuits_val": int(data["n_circuits_val"]),
        }
