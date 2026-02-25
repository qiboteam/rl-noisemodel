"""Unit tests for reward functions."""

import pytest
import numpy as np

from rlnoise.config import RewardConfig
from rlnoise.reward import (
    RewardFunction,
    create_reward_function,
    mse,
    mae,
    trace_distance,
    compute_fidelity,
)


class TestDistanceMetrics:
    """Test distance metric functions."""
    
    @pytest.fixture
    def identity_dm(self):
        """Identity density matrix (1-qubit)."""
        dm = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        return dm
    
    @pytest.fixture
    def mixed_dm(self):
        """Maximally mixed density matrix (1-qubit)."""
        dm = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=complex)
        return dm
    
    def test_mse_identical(self, identity_dm):
        """Test MSE is zero for identical matrices."""
        distance = mse(identity_dm, identity_dm)
        assert np.isclose(distance, 0.0)
    
    def test_mse_different(self, identity_dm, mixed_dm):
        """Test MSE is positive for different matrices."""
        distance = mse(identity_dm, mixed_dm)
        assert distance > 0.0
    
    def test_mae_identical(self, identity_dm):
        """Test MAE is zero for identical matrices."""
        distance = mae(identity_dm, identity_dm)
        assert np.isclose(distance, 0.0)
    
    def test_mae_different(self, identity_dm, mixed_dm):
        """Test MAE is positive for different matrices."""
        distance = mae(identity_dm, mixed_dm)
        assert distance > 0.0
    
    def test_trace_distance_identical(self, identity_dm):
        """Test trace distance is zero for identical matrices."""
        distance = trace_distance(identity_dm, identity_dm)
        assert np.isclose(distance, 0.0, atol=1e-10)
    
    def test_trace_distance_bounds(self, identity_dm, mixed_dm):
        """Test trace distance is between 0 and 1."""
        distance = trace_distance(identity_dm, mixed_dm)
        assert 0.0 <= distance <= 1.0
    
    def test_fidelity_identical(self, identity_dm):
        """Test fidelity distance is zero for identical matrices."""
        distance = compute_fidelity(identity_dm, identity_dm)
        assert np.isclose(distance, 0.0, atol=1e-10)
    
    def test_fidelity_bounds(self, identity_dm, mixed_dm):
        """Test fidelity distance is between 0 and 1."""
        distance = compute_fidelity(identity_dm, mixed_dm)
        assert 0.0 <= distance <= 1.0


class TestRewardConfig:
    """Test RewardConfig validation."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = RewardConfig()
        assert config.metric == "trace"
        assert config.function == "inverted_squared"
        assert config.alpha == 20.0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = RewardConfig(
            metric="mse",
            function="linear",
            alpha=10.0
        )
        assert config.metric == "mse"
        assert config.function == "linear"
        assert config.alpha == 10.0
    
    def test_invalid_metric(self):
        """Test that invalid metric raises error."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            RewardConfig(metric="invalid")
    
    def test_invalid_function(self):
        """Test that invalid function raises error."""
        with pytest.raises(Exception):
            RewardConfig(function="invalid")
    
    def test_invalid_alpha(self):
        """Test that negative alpha raises error."""
        with pytest.raises(Exception):
            RewardConfig(alpha=-1.0)


class TestRewardFunction:
    """Test RewardFunction class."""
    
    @pytest.fixture
    def identity_dm(self):
        """Identity density matrix."""
        return np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    
    @pytest.fixture
    def mixed_dm(self):
        """Mixed density matrix."""
        return np.array([[0.5, 0.0], [0.0, 0.5]], dtype=complex)
    
    def test_initialization(self):
        """Test reward function initialization."""
        config = RewardConfig(metric="trace", function="inverted", alpha=10.0)
        reward_fn = RewardFunction(config)
        
        assert reward_fn.config == config
        assert reward_fn.metric is not None
        assert reward_fn.transform is not None
    
    def test_non_terminal_reward(self, identity_dm):
        """Test that non-terminal states give zero reward."""
        config = RewardConfig()
        reward_fn = RewardFunction(config)
        
        reward = reward_fn(identity_dm, identity_dm, is_terminal=False)
        assert reward == 0.0
    
    def test_terminal_reward_identical(self, identity_dm):
        """Test reward for identical matrices is high."""
        config = RewardConfig(metric="trace", function="inverted_squared", alpha=20.0)
        reward_fn = RewardFunction(config)
        
        reward = reward_fn(identity_dm, identity_dm, is_terminal=True)
        # Distance is ~0, so 1/(20*0^2 + eps) should be very large
        assert reward > 1e6
    
    def test_terminal_reward_different(self, identity_dm, mixed_dm):
        """Test reward for different matrices is lower."""
        config = RewardConfig(metric="trace", function="inverted_squared", alpha=20.0)
        reward_fn = RewardFunction(config)
        
        reward = reward_fn(identity_dm, mixed_dm, is_terminal=True)
        # Should be positive but not as large
        assert reward > 0.0
        assert reward < 1e6
    
    def test_linear_transformation(self, identity_dm, mixed_dm):
        """Test linear transformation function."""
        config = RewardConfig(metric="mse", function="linear", alpha=1.0)
        reward_fn = RewardFunction(config)
        
        reward = reward_fn(identity_dm, mixed_dm, is_terminal=True)
        # Linear should give negative reward for distance
        assert reward < 0.0
    
    def test_log_transformation(self, identity_dm, mixed_dm):
        """Test log transformation function."""
        config = RewardConfig(metric="mse", function="log", alpha=1.0)
        reward_fn = RewardFunction(config)
        
        reward = reward_fn(identity_dm, mixed_dm, is_terminal=True)
        # Log should give negative reward
        assert reward < 0.0
    
    def test_evaluate_metrics(self, identity_dm, mixed_dm):
        """Test evaluate method returns all metrics."""
        config = RewardConfig(metric="trace")
        reward_fn = RewardFunction(config)
        
        metrics = reward_fn.evaluate(identity_dm, mixed_dm)
        
        # Check all expected keys are present
        assert "mse" in metrics
        assert "mae" in metrics
        assert "trace_distance" in metrics
        assert "fidelity" in metrics
        assert "distance" in metrics
        assert "reward" in metrics
        assert "metric_used" in metrics
        assert "function_used" in metrics
        
        # Check values are reasonable
        assert metrics["mse"] >= 0.0
        assert metrics["mae"] >= 0.0
        assert 0.0 <= metrics["trace_distance"] <= 1.0
        assert 0.0 <= metrics["fidelity"] <= 1.0
    
    def test_different_metrics(self, identity_dm, mixed_dm):
        """Test that different metrics give different rewards."""
        metrics = ["mse", "trace", "fidelity", "mae"]
        rewards = []
        
        for metric in metrics:
            config = RewardConfig(metric=metric, function="inverted", alpha=1.0)
            reward_fn = RewardFunction(config)
            reward = reward_fn(identity_dm, mixed_dm, is_terminal=True)
            rewards.append(reward)
        
        # At least some rewards should be different
        assert len(set(rewards)) > 1


class TestCreateRewardFunction:
    """Test convenience function."""
    
    def test_create_with_defaults(self):
        """Test creating reward function with defaults."""
        reward_fn = create_reward_function()
        
        assert isinstance(reward_fn, RewardFunction)
        assert reward_fn.config.metric == "trace"
        assert reward_fn.config.function == "inverted_squared"
        assert reward_fn.config.alpha == 20.0
    
    def test_create_with_custom_params(self):
        """Test creating reward function with custom parameters."""
        reward_fn = create_reward_function(
            metric="mse",
            function="linear",
            alpha=5.0
        )
        
        assert reward_fn.config.metric == "mse"
        assert reward_fn.config.function == "linear"
        assert reward_fn.config.alpha == 5.0
    
    def test_created_function_works(self):
        """Test that created function computes rewards."""
        reward_fn = create_reward_function()
        
        dm1 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        dm2 = np.array([[0.9, 0.1], [0.1, 0.1]], dtype=complex)
        
        reward = reward_fn(dm1, dm2, is_terminal=True)
        assert isinstance(reward, float)
        assert reward > 0.0
