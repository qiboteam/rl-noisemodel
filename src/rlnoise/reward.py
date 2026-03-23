"""Reward functions for reinforcement learning environments."""

from typing import Callable
import numpy as np
from qibo.quantum_info import fidelity as qibo_fidelity

from rlnoise.config import RewardConfig


def mse(dm1: np.ndarray, dm2: np.ndarray) -> float:
    """Mean squared error between two density matrices.
    
    Args:
        dm1: First density matrix
        dm2: Second density matrix
        
    Returns:
        MSE distance
    """
    return np.mean(np.abs(dm1 - dm2) ** 2)


def mae(dm1: np.ndarray, dm2: np.ndarray) -> float:
    """Mean absolute error between two density matrices.
    
    Args:
        dm1: First density matrix
        dm2: Second density matrix
        
    Returns:
        MAE distance
    """
    return np.mean(np.abs(dm1 - dm2))


def trace_distance(dm1: np.ndarray, dm2: np.ndarray) -> float:
    """Trace distance between two density matrices.
    
    The trace distance is defined as: (1/2) * Tr(|dm1 - dm2|)
    where |A| = sqrt(A† A) is the matrix absolute value.
    
    Args:
        dm1: First density matrix
        dm2: Second density matrix
        
    Returns:
        Trace distance (between 0 and 1)
    """
    diff = dm1 - dm2
    # Compute eigenvalues of the difference
    eigenvalues = np.linalg.eigvalsh(diff @ diff.conj().T)
    # Take square root of eigenvalues and sum
    return 0.5 * np.sum(np.sqrt(np.abs(eigenvalues)))


def compute_fidelity(dm1: np.ndarray, dm2: np.ndarray) -> float:
    """Quantum fidelity between two density matrices.
    
    Uses Qibo's implementation for accurate fidelity calculation.
    Returns 1 - fidelity to be used as a distance metric.
    
    Args:
        dm1: First density matrix
        dm2: Second density matrix
        
    Returns:
        1 - fidelity (distance, 0 = identical, 1 = orthogonal)
    """
    fid = qibo_fidelity(dm1, dm2)
    return 1.0 - fid


class RewardFunction:
    """Flexible reward function for quantum circuit environments.
    
    This class combines a distance metric with a transformation function
    to compute rewards for reinforcement learning.
    
    Args:
        config: RewardConfig specifying metric and function type
    
    Example:
        >>> config = RewardConfig(metric="trace", function="inverted_squared", alpha=20.0)
        >>> reward_fn = RewardFunction(config)
        >>> reward = reward_fn(noisy_dm, target_dm, is_terminal=True)
    """
    
    def __init__(self, config: RewardConfig):
        self.config = config
        
        # Select distance metric
        self.metric = self._get_metric_function(config.metric)
        
        # Select transformation function
        self.transform = self._get_transform_function(config.function, config.alpha)
    
    def _get_metric_function(self, metric: str) -> Callable:
        """Get the distance metric function.
        
        Args:
            metric: Name of metric ('mse', 'fidelity', 'trace', 'mae')
            
        Returns:
            Metric function
        """
        metric_map = {
            "mse": mse,
            "fidelity": compute_fidelity,
            "trace": trace_distance,
            "mae": mae,
        }
        
        if metric not in metric_map:
            raise ValueError(f"Unknown metric: {metric}")
        
        return metric_map[metric]
    
    def _get_transform_function(self, function: str, alpha: float) -> Callable:
        """Get the reward transformation function.
        
        Args:
            function: Type of transformation
            alpha: Scaling parameter
            
        Returns:
            Transform function
        """
        transform_map = {
            "log": lambda x: -np.log(alpha * x + 1e-15),
            "linear": lambda x: -alpha * x,  # Negative for reward
            "inverted": lambda x: 1.0 / (alpha * x + 1e-15),
            "inverted_squared": lambda x: 1.0 / (alpha * x**2 + 1e-10),
        }
        
        if function not in transform_map:
            raise ValueError(f"Unknown function: {function}")
        
        return transform_map[function]
    
    def __call__(
        self, 
        predicted_dm: np.ndarray, 
        target_dm: np.ndarray, 
        is_terminal: bool = False
    ) -> float:
        """Compute reward based on density matrix comparison.
        
        Args:
            predicted_dm: Density matrix from noisy circuit
            target_dm: Target density matrix
            is_terminal: Whether this is the final step (only compute reward at end)
            
        Returns:
            Reward value (higher is better)
        """
        if not is_terminal:
            return 0.0
        
        # Compute distance
        distance = self.metric(predicted_dm, target_dm)
        
        # Transform to reward (higher is better)
        reward = self.transform(distance)
        
        return float(reward)
    
    def evaluate(self, predicted_dm: np.ndarray, target_dm: np.ndarray) -> dict:
        """Evaluate all metrics for a density matrix pair.
        
        Useful for analysis and debugging.
        
        Args:
            predicted_dm: Predicted density matrix
            target_dm: Target density matrix
            
        Returns:
            Dictionary with all metric values and reward
        """
        metrics = {
            "mse": mse(predicted_dm, target_dm),
            "mae": mae(predicted_dm, target_dm),
            "trace_distance": trace_distance(predicted_dm, target_dm),
            "fidelity": 1.0 - compute_fidelity(predicted_dm, target_dm),  # Convert back to fidelity
        }
        
        return metrics


# Convenience function
def create_reward_function(
    metric: str = "trace",
    function: str = "inverted_squared",
    alpha: float = 20.0
) -> RewardFunction:
    """Create a reward function with given parameters.
    
    Args:
        metric: Distance metric to use
        function: Transformation function type
        alpha: Scaling parameter
        
    Returns:
        Configured RewardFunction
    """
    config = RewardConfig(metric=metric, function=function, alpha=alpha)
    return RewardFunction(config)
