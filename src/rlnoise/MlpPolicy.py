import torch
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np


class MlPFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(
            self,
            observation_space,
            features_dim,
               
    ):
        super().__init__(observation_space, features_dim)

        sample = torch.as_tensor(observation_space.sample()[None]).float().flatten(1,-1)
        in_dim=sample.shape[-1]
        print('indim: ',in_dim)

        self.Linear1=torch.nn.Sequential(torch.nn.Linear(in_features=in_dim,out_features=128),torch.nn.Tanh(),torch.nn.AvgPool1d(kernel_size=4,stride=2))
        with torch.no_grad():
            #shape = torch.nn.functional.avg_pool2d(conv1(sample),kernel_size=filter_shape).shape
            in_feature2 = self.Linear1(sample).shape[-1]
            print('in_feature2: ',in_feature2)
        self.Linear2=torch.nn.Linear(in_features=in_feature2,out_features=64)

        self.Linear3=torch.nn.Linear(in_features=64,out_features=features_dim)



        self.Mlp = torch.nn.Sequential(
            torch.nn.Flatten(1,-1),
            self.Linear1,

            self.Linear2,
            torch.nn.Tanh(),
            self.Linear3,
            torch.nn.ELU(),


            
        )


    def forward(self, x):
        #test=np.array(torch.Tensor.cpu(x[0]))
        #print('input of feature extractor: \n',test.transpose(2,1,0))
        feature = self.Mlp(x)
        
        return feature