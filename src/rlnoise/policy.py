import torch
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CNNFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(
            self,
            observation_space,
            features_dim,
            filter_shape = (4,4),
            n_filters = 32                 
    ):
        super().__init__(observation_space, features_dim)
        indim = observation_space.shape[0]
        sample = torch.as_tensor(observation_space.sample()[None]).float()

        conv1 = torch.nn.Conv2d(1, n_filters, filter_shape)

        # Compute shape by doing one forward pass
        with torch.no_grad():
            shape = conv1(sample).shape

        conv2 = torch.nn.Conv2d(n_filters, n_filters, (max(1, int(shape[1]/2)), shape[2]))
        
        self.cnn = torch.nn.Sequential(
            conv1,
            torch.nn.ReLU(),
            conv2,
            torch.nn.ReLU(),
            torch.nn.Flatten(0,-1),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            hdim = self.cnn(sample).shape[0]

        assert hdim > features_dim

        self.linear = torch.nn.Sequential(torch.nn.Linear(hdim, features_dim), torch.nn.ReLU())

    def forward(self, x):
        return self.linear(self.cnn(x))


