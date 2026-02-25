import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CNNFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(
            self,
            observation_space,
            features_dim,
            filter_shape,
            n_filters,                 
    ):
        super().__init__(observation_space, features_dim)
        indim = observation_space.shape[0]
        sample = torch.as_tensor(observation_space.sample()[None]).float()
        conv1 = torch.nn.Conv2d( in_channels=indim, out_channels=n_filters, 
                                kernel_size=filter_shape, padding=0)
        # conv2 = torch.nn.Conv2d( in_channels=n_filters, out_channels=n_filters, 
        #                         kernel_size=filter_shape)
        
        self.cnn = torch.nn.Sequential(
            conv1,
            torch.nn.ReLU(),
            # conv2,
            # torch.nn.ReLU(),
            torch.nn.Flatten(1,-1),
        )
        with torch.no_grad():
            hdim = self.cnn(sample).shape[-1]

        self.linear = torch.nn.Sequential(torch.nn.Linear(hdim, features_dim), torch.nn.ReLU())

    def forward(self, x):
        x = self.cnn(x)  
        return self.linear(x)