import json
import os
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from qibo import gates
from rlnoise.metrics import trace_distance,compute_fidelity

current_path = Path(__file__).parent
with open(f"{current_path}/config.json") as f:
    config = json.load(f)

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
    def __call__(self, circuit, target, final=False,alpha=1.):
        reward_type=config['reward']['reward_type']
        if final:
            circuit_dm=np.array(circuit().state())
            if reward_type in ["log", "Log"]:
                dm_mse=alpha*self.metric(circuit_dm, target)
                reward = -np.log(dm_mse) if -np.log(dm_mse) < 1000 else 1000.
            elif reward_type=="mse":
                reward=1-alpha*self.metric(circuit_dm, target)

            elif reward_type in ["trace_distance", "trace distance"]:
                reward=-np.log(trace_distance(circuit_dm,target))

            elif reward_type.lower()=="mixed":
                reward=compute_fidelity(circuit_dm, target)*(1-5*self.metric(circuit_dm, target))*(1-trace_distance(circuit_dm, target))

            elif reward_type.lower()=="fidelity":
                fidelity = compute_fidelity(circuit_dm, target)
                if -np.log(1-fidelity) < 1000:
                    reward=-np.log(1-fidelity) 
                else:
                    reward=1000.


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

#Add Bures distance as metric to evaluate the performance