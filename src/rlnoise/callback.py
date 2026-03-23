"""Training callbacks for monitoring and evaluation."""

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
    ):
        """Initialize the training callback.
        
        Args:
            env: Environment for evaluation
            check_freq: Steps between evaluations
            save_path: Path to save best model (without extension)
            save_best: Whether to save best model
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        self.env = env
        self.check_freq = check_freq
        self.save_path = save_path
        self.save_best = save_best
        
        # Initialize tracking
        self.best_mean_reward = -np.inf
        self.eval_results = []
        self.train_results = []
        self.timestep_list = []
        
        # Create save directory if needed
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _on_step(self) -> bool:
        """Called after each step during training.
        
        Returns:
            bool: If False, training stops
        """
        # Check if it's time to evaluate
        if self.n_calls % self.check_freq == 0:
            self._evaluate()
        
        return True
    
    def _evaluate(self):
        """Evaluate model on training and validation sets."""
        if self.verbose > 0:
            print(f"\n{'='*60}")
            print(f"Evaluation at timestep {self.num_timesteps}")
            print(f"{'='*60}")
        
        # Evaluate on training set
        train_metrics = self._evaluate_on_set(train=True)
        self.train_results.append(train_metrics)
        
        # Evaluate on validation set (if exists)
        if self.env.n_circuits > self.env.n_circuits_train:
            val_metrics = self._evaluate_on_set(train=False)
            self.eval_results.append(val_metrics)
        else:
            val_metrics = None
        
        # Store timestep
        self.timestep_list.append(self.num_timesteps)
        
        # Print results
        if self.verbose > 0:
            self._print_metrics("Training", train_metrics)
            if val_metrics is not None:
                self._print_metrics("Validation", val_metrics)
        
        # Save best model based on validation reward (or training if no val set)
        if self.save_best and self.save_path is not None:
            mean_reward = val_metrics[0] if val_metrics is not None else train_metrics[0]
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print(f"\nNew best model! Mean reward: {mean_reward:.4f}")
                    print(f"Saving to {self.save_path}")
                self.model.save(self.save_path)
        
        if self.verbose > 0:
            print(f"{'='*60}\n")
    
    def _evaluate_on_set(self, train: bool = True) -> np.ndarray:
        """Evaluate model on training or validation set.
        
        Args:
            train: If True, evaluate on training set, else validation
        
        Returns:
            Array of metrics: [mean_reward, std_reward]
        """
        if train:
            start = 0
            stop = self.env.n_circuits_train
        else:
            start = self.env.n_circuits_train
            stop = self.env.n_circuits
        
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
        """Print evaluation metrics.
        
        Args:
            set_name: Name of the set ("Training" or "Validation")
            metrics: Array of metrics [mean_reward, std_reward]
        """
        print(f"{set_name} Set:")
        print(f"  Reward: {metrics[0]:.4f} ± {metrics[1]:.4f}")
    
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
