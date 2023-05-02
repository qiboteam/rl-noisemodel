import torch
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np


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
        print('observation spac shape: ',observation_space.shape)
        sample = torch.as_tensor(observation_space.sample()[None]).float()
        #print('indim: ',indim)
        #print('filter shape: ',filter_shape)
        filter_shape=(1,2)
        conv1 = torch.nn.Conv2d( in_channels=indim,out_channels=64, kernel_size=filter_shape) #adding pooling layer?
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            #shape = torch.nn.functional.avg_pool2d(conv1(sample),kernel_size=filter_shape).shape
            shape = conv1(sample).shape
            
        #conv2 = torch.nn.Conv2d(n_filters, n_filters, (max(int(shape[2]/2), 1), shape[3]))  #aggiungi linear layer prima di conv1
        conv2 = torch.nn.Conv2d(64, 32, (max(int(shape[2]/2), 1), shape[3]))

        self.cnn = torch.nn.Sequential(
            conv1,
            torch.nn.ReLU(), # Relu might not be great if we have negative angles, ELU
            #torch.nn.AvgPool2d(kernel_size=filter_shape),
            conv2,
            torch.nn.ReLU(),
            torch.nn.Flatten(1,-1),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            hdim = self.cnn(sample).shape[-1]

        if hdim < features_dim:
            print(f'Warning, using features_dim ({features_dim}) greater than hidden dim ({hdim}).')

        self.linear = torch.nn.Sequential(torch.nn.Linear(hdim, features_dim), torch.nn.ELU())

    def forward(self, x):
        #test=np.array(torch.Tensor.cpu(x[0]))
        #print('input of feature extractor: \n',test.transpose(2,1,0))
        x = self.cnn(x)
        
        return self.linear(x)