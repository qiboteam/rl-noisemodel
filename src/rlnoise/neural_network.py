"""Neural network feature extractors for RL agents."""

import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CNNFeaturesExtractor(BaseFeaturesExtractor):
    """CNN feature extractor for quantum circuit observations.

    Extracts features from the sliding window observations of quantum circuits
    using convolutional layers followed by a linear layer.

    Architecture:
        Conv2D -> ReLU -> Flatten -> Linear -> ReLU

    Args:
        observation_space: Gymnasium Box space for observations
        features_dim: Output dimension of the feature extractor
        filter_shape: Tuple (height, width) for convolutional filter
        n_filters: Number of convolutional filters

    Example:
        >>> from gymnasium import spaces
        >>> obs_space = spaces.Box(0, 1, shape=(8, 3, 3), dtype=np.float32)
        >>> extractor = CNNFeaturesExtractor(
        ...     observation_space=obs_space,
        ...     features_dim=64,
        ...     filter_shape=(3, 3),
        ...     n_filters=32
        ... )
    """

    def __init__(
        self,
        observation_space,
        features_dim: int,
        filter_shape: tuple,
        n_filters: int,
    ):
        """Initialize the CNN feature extractor.

        Args:
            observation_space: Gymnasium observation space
            features_dim: Dimension of extracted features
            filter_shape: (height, width) of convolutional kernel
            n_filters: Number of convolutional filters
        """
        super().__init__(observation_space, features_dim)

        # Get input dimension from observation space
        # Shape is (encoding_dim, n_qubits, kernel_size)
        indim = observation_space.shape[0]

        # Create convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=indim,
            out_channels=n_filters,
            kernel_size=filter_shape,
            padding=0
        )

        # Build CNN pipeline
        self.cnn = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.Flatten(1, -1),
        )

        # Calculate output dimension of CNN by passing a sample through
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            hdim = self.cnn(sample).shape[-1]

        # Linear layer to project to features_dim
        self.linear = nn.Sequential(
            nn.Linear(hdim, features_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feature extractor.

        Args:
            x: Input tensor of shape (batch, encoding_dim, n_qubits, kernel_size)

        Returns:
            Feature tensor of shape (batch, features_dim)
        """
        x = self.cnn(x)
        return self.linear(x)
