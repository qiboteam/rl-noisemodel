"""Tests for neural network feature extractors."""

import pytest
import numpy as np
import torch
from gymnasium import spaces

from rlnoise.neural_network import CNNFeaturesExtractor


class TestCNNFeaturesExtractor:
    """Test suite for CNN feature extractor."""
    
    @pytest.fixture
    def observation_space(self):
        """Create a test observation space."""
        # Shape: (encoding_dim, n_qubits, kernel_size)
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(8, 3, 3),
            dtype=np.float32
        )
    
    def test_initialization(self, observation_space):
        """Test basic initialization."""
        extractor = CNNFeaturesExtractor(
            observation_space=observation_space,
            features_dim=64,
            filter_shape=(3, 3),
            n_filters=32
        )
        
        assert extractor.features_dim == 64
        assert isinstance(extractor.conv1, torch.nn.Conv2d)
        assert isinstance(extractor.cnn, torch.nn.Sequential)
        assert isinstance(extractor.linear, torch.nn.Sequential)
    
    def test_forward_pass(self, observation_space):
        """Test forward pass produces correct output shape."""
        extractor = CNNFeaturesExtractor(
            observation_space=observation_space,
            features_dim=64,
            filter_shape=(3, 3),
            n_filters=32
        )
        
        # Create batch of observations
        batch_size = 4
        obs = torch.randn(batch_size, 8, 3, 3)
        
        # Forward pass
        features = extractor(obs)
        
        # Check output shape
        assert features.shape == (batch_size, 64)
        assert not torch.isnan(features).any()
        assert torch.isfinite(features).all()
    
    def test_single_observation(self, observation_space):
        """Test with single observation."""
        extractor = CNNFeaturesExtractor(
            observation_space=observation_space,
            features_dim=32,
            filter_shape=(3, 2),
            n_filters=16
        )
        
        # Single observation
        obs = torch.randn(1, 8, 3, 3)
        features = extractor(obs)
        
        assert features.shape == (1, 32)
    
    def test_different_filter_sizes(self, observation_space):
        """Test with different filter configurations."""
        filter_configs = [
            ((3, 1), 16),
            ((3, 2), 24),
            ((3, 3), 32),
        ]
        
        for filter_shape, n_filters in filter_configs:
            extractor = CNNFeaturesExtractor(
                observation_space=observation_space,
                features_dim=48,
                filter_shape=filter_shape,
                n_filters=n_filters
            )
            
            obs = torch.randn(2, 8, 3, 3)
            features = extractor(obs)
            
            assert features.shape == (2, 48)
    
    def test_different_feature_dims(self, observation_space):
        """Test with different feature dimensions."""
        for features_dim in [16, 32, 64, 128]:
            extractor = CNNFeaturesExtractor(
                observation_space=observation_space,
                features_dim=features_dim,
                filter_shape=(3, 2),
                n_filters=16
            )
            
            obs = torch.randn(3, 8, 3, 3)
            features = extractor(obs)
            
            assert features.shape == (3, features_dim)
    
    def test_gradient_flow(self, observation_space):
        """Test that gradients flow through the network."""
        extractor = CNNFeaturesExtractor(
            observation_space=observation_space,
            features_dim=64,
            filter_shape=(3, 3),
            n_filters=32
        )
        
        # Create observation with gradient tracking
        obs = torch.randn(2, 8, 3, 3, requires_grad=True)
        
        # Forward pass
        features = extractor(obs)
        
        # Compute loss and backward
        loss = features.sum()
        loss.backward()
        
        # Check gradients exist
        assert obs.grad is not None
        assert not torch.isnan(obs.grad).any()
    
    def test_deterministic_output(self, observation_space):
        """Test that same input produces same output."""
        extractor = CNNFeaturesExtractor(
            observation_space=observation_space,
            features_dim=64,
            filter_shape=(3, 2),
            n_filters=16
        )
        
        extractor.eval()  # Set to eval mode
        
        obs = torch.randn(1, 8, 3, 3)
        
        # Forward pass twice
        with torch.no_grad():
            features1 = extractor(obs)
            features2 = extractor(obs)
        
        # Should be identical
        assert torch.allclose(features1, features2)
    
    def test_with_different_observation_spaces(self):
        """Test with various observation space configurations."""
        configs = [
            (8, 1, 3),   # 1 qubit
            (8, 2, 3),   # 2 qubits
            (8, 3, 5),   # different kernel size
            (12, 2, 3),  # different encoding dim
        ]
        
        for encoding_dim, n_qubits, kernel_size in configs:
            obs_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(encoding_dim, n_qubits, kernel_size),
                dtype=np.float32
            )
            
            extractor = CNNFeaturesExtractor(
                observation_space=obs_space,
                features_dim=64,
                filter_shape=(n_qubits, 2),
                n_filters=32
            )
            
            obs = torch.randn(2, encoding_dim, n_qubits, kernel_size)
            features = extractor(obs)
            
            assert features.shape == (2, 64)
    
    def test_batch_independence(self, observation_space):
        """Test that batch samples are processed independently."""
        extractor = CNNFeaturesExtractor(
            observation_space=observation_space,
            features_dim=64,
            filter_shape=(3, 2),
            n_filters=16
        )
        
        extractor.eval()
        
        # Create two different observations
        obs1 = torch.randn(1, 8, 3, 3)
        obs2 = torch.randn(1, 8, 3, 3)
        obs_batch = torch.cat([obs1, obs2], dim=0)
        
        with torch.no_grad():
            # Process individually
            features1 = extractor(obs1)
            features2 = extractor(obs2)
            
            # Process as batch
            features_batch = extractor(obs_batch)
        
        # Results should match
        assert torch.allclose(features_batch[0], features1[0])
        assert torch.allclose(features_batch[1], features2[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
