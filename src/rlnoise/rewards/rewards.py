from abc import ABC, abstractmethod
import numpy as np
from qibo import gates
from qibo.quantum_info import trace_distance
from configparser import ConfigParser

params=ConfigParser()
params.read("src/rlnoise/config.ini")
class Reward(ABC):

    def __init__(self, metric=lambda x,y: np.sqrt(np.abs(((x-y)**2)).mean())):
        self.metric = metric

    @abstractmethod
    def __call__(self, circuit, target, final=False): #target sono le labels dello stato i-esimo
        pass

    
class FrequencyReward(Reward):

    def __call__(self, circuit, target, final=False):
        if final:
            circuit.add(gates.M(*range(circuit.nqubits)))
            # get the number of shots
            nshots = 0
            for v in target.values():
                nshots += v
            # normalize in the number of shots
            target = {k: v/nshots for k,v in target.items()}
            # get the predicted statistics
            freq = circuit(nshots=nshots).frequencies()         
            freq = {k: v/nshots for k,v in freq.items()}
            # Fill the missing keys
            for k in freq.keys() | target.keys():
                if k not in freq:
                    freq[k] = 0
                elif k not in target:
                    target[k] = 0
        
            freq = np.array([ freq[k] for k in target.keys() ])
            target = np.array([ target[k] for k in target.keys() ])
            reward = 1 - self.metric(freq, target)
        else:
            reward = 0
        return reward
    
class DensityMatrixReward(Reward):
    def __call__(self, circuit, target, final=False,alpha=10.):
        reward_type=params.get('reward','reward_type')
        if final:
            circuit_dm=np.array(circuit().state())
            dm_mse=alpha*self.metric(circuit_dm, target)
            #dm_mae=(np.abs(circuit_dm-target)).mean()
            #if dm_mse<0.2:
            #    reward = (1 - dm_mse)*10
            #else:
            #    reward = (1 - dm_mse)
            
            #print('log(mse): ',np.log(dm_mse))
            '''
            if dm_mse < 0.2:
                reward=-np.log(dm_mse)
            else:
                reward=0.
            '''
            if reward_type=="log" or reward_type=="Log":
                if -np.log(dm_mse) < 1000:
                    reward=-np.log(dm_mse) #mae or exp
                else:
                    reward=1000.
            elif reward_type=="mse":
                reward=1-dm_mse
            #reward=1-trace_distance(circuit_dm, target)
            #print('reward: ',reward)
            #print('\n dm MSE: ',dm_mse )
            #print('Trace distance: ',self.trace_distance(circuit_dm, target))
           # print('dm MAE',dm_mae)
        else:
            reward = 0.
        return reward  #other possible metric to evaluate distance between DMs is Bures distance. See https://arxiv.org/pdf/2105.02743.pdf
    


    


if __name__ == '__main__':

    from qibo.models import Circuit
    from qibo import gates

    c = Circuit(1)
    c.add(gates.X(0))
    c.add(gates.M(0))
    target = {'0': 0, '1': 100}
    r = FrequencyReward()
    print(r(c,target))
