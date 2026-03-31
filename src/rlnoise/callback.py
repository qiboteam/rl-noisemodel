"""Training callbacks for monitoring and evaluation."""

import sys
import numpy as np
from pathlib import Path
from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback

from rlnoise.gym_env import QuantumCircuitEnv
from rlnoise.reward import RewardFunction


class TrainingCallback(BaseCallback):
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
    
    def __init__(
        self,
        env: QuantumCircuitEnv,
        check_freq: int = 1000,
        save_path: Optional[str] = None,
        save_best: bool = True,
        verbose: int = 1,
        out_stream=None,
    ):
        """Initialize the training callback.
        
        Args:
            env: Environment for evaluation
            check_freq: Steps between evaluations
            save_path: Path to save best model (without extension)
            save_best: Whether to save best model
            verbose: Verbosity level
            out_stream: Output stream for prints (captured before rich wraps stdout)
        """
        super().__init__(verbose)
        
        self.env = env
        self.check_freq = check_freq
        self.save_path = save_path
        self.save_best = save_best
        self._out = out_stream if out_stream is not None else sys.stdout
        self.has_val_set = env.n_circuits > env.n_circuits_train
        
        # Initialize tracking
        self.best_mean_reward = -np.inf
        self.eval_results = []
        self.train_results = []
        self.timestep_list = []
        
        # Accumulate episode-terminal rewards from the live rollout between evals
        self._rollout_rewards: list = []
        
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
        # Collect terminal rewards from the live rollout (reward is non-zero only at done)
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])
        for done, reward in zip(dones, rewards):
            if done:
                self._rollout_rewards.append(float(reward))

        if self.n_calls % self.check_freq == 0:
            self._evaluate()
        
        return True
    
    def _evaluate(self):
        """Record training reward from rollout and evaluate validation set."""
        # Training reward: average of terminal rewards collected since last eval
        if self._rollout_rewards:
            r = np.array(self._rollout_rewards)
            train_metrics = np.array([r.mean(), r.std()])
        else:
            train_metrics = np.zeros(2)
        self._rollout_rewards = []  # reset for next interval
        self.train_results.append(train_metrics)
        
        # Evaluate on validation set, or store zeros if none exists
        if self.has_val_set:
            val_metrics = self._evaluate_on_set(train=False)
        else:
            val_metrics = np.zeros(2)
        self.eval_results.append(val_metrics)
        
        # Store timestep
        self.timestep_list.append(self.num_timesteps)
        
        # Print single-line summary
        if self.verbose > 0:
            msg = (
                f"Step {self.num_timesteps:>7d} | "
                f"Train reward: {train_metrics[0]:.4f} ± {train_metrics[1]:.4f}  |  "
                f"Val reward: {val_metrics[0]:.4f} ± {val_metrics[1]:.4f}"
            )
            print(msg, file=self._out, flush=True)
        
        # Save best model based on validation reward (or training if no val set)
        if self.save_best and self.save_path is not None:
            mean_reward = val_metrics[0] if self.has_val_set else train_metrics[0]
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.save_path)
    
    def _evaluate_on_set(self, train: bool = True) -> np.ndarray:
        """Run deterministic episodes over the validation set.
        
        Returns:
            Array [mean_reward, std_reward]
        """
        start = 0 if train else self.env.n_circuits_train
        stop = self.env.n_circuits_train if train else self.env.n_circuits
        
        rewards = []
        
        for i in range(start, stop):
            # Reset to specific circuit
            obs, _ = self.env.reset(options={"circuit_idx": i})
            done = False
            
            while not done:
                # Get deterministic action from model
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)
            
            rewards.append(reward)
        
        rewards = np.array(rewards)
        return np.array([rewards.mean(), rewards.std()])
    
    def _print_metrics(self, set_name: str, metrics: np.ndarray):
        """Print evaluation metrics."""
        print(f"{set_name}: reward {metrics[0]:.4f} ± {metrics[1]:.4f}", file=self._out, flush=True)
    
    def get_results(self) -> dict:
        """Get all recorded results.
        
        Returns:
            Dictionary with timesteps and results for train/val sets
        """
        return {
            "timesteps": self.timestep_list,
            "train_results": self.train_results,
            "eval_results": self.eval_results,
            "best_mean_reward": self.best_mean_reward,
        }
