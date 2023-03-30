from abc import ABC, abstractmethod
import numpy as np
from qibo import gates


class Reward(ABC):

    def __init__(self, metric=lambda x,y: np.sqrt((x-y)**2).mean()):
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
    def __call__(self, circuit, target, final=False):
        if final:
            circuit_dm=np.array(circuit().state())
            #print('Density matrix circuito',circuit_dm)
            #print('Density matrix target', target)
            reward = 1 - self.metric(circuit_dm, target)
            
        else:
            reward = 0
        #print('La reward e`',reward)
        return reward
    


if __name__ == '__main__':

    from qibo.models import Circuit
    from qibo import gates

    c = Circuit(1)
    c.add(gates.X(0))
    c.add(gates.M(0))
    target = {'0': 0, '1': 100}
    r = FrequencyReward()
    print(r(c,target))
