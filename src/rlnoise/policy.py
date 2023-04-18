import torch
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CNNFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(
            self,
            observation_space,
            features_dim,
            filter_shape,
            n_filters = 32                 
    ):
        super().__init__(observation_space, features_dim)
        indim = observation_space.shape[0]
        sample = torch.as_tensor(observation_space.sample()[None]).float()
        #print('indim: ',indim)
        #print('filter shape: ',filter_shape)
        conv1 = torch.nn.Conv2d( in_channels=indim,out_channels=n_filters, kernel_size=filter_shape)
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            shape = conv1(sample).shape
            
        conv2 = torch.nn.Conv2d(n_filters, n_filters, (max(int(shape[2]/2), 1), shape[3]))
        
        self.cnn = torch.nn.Sequential(
            conv1,
            torch.nn.ELU(), # Relu might not be great if we have negative angles
            conv2,
            torch.nn.ELU(),
            torch.nn.Flatten(1,-1),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            hdim = self.cnn(sample).shape[-1]

        if hdim < features_dim:
            print(f'Warning, using features_dim ({features_dim}) greater than hidden dim ({hdim}).')

        self.linear = torch.nn.Sequential(torch.nn.Linear(hdim, features_dim), torch.nn.ELU())

    def forward(self, x):
        #print(x.shape)
        x = self.cnn(x)
        
        return self.linear(x)