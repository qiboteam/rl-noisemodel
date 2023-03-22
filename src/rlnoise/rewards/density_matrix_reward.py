from qibo import gates
from qibo.models import Circuit
import numpy as np

def dm_reward_stablebaselines(circuit, label):
    generated_dm = np.asarray(circuit().state())
    return compute_reward(generated_dm, label)

def step_reward_stablebaselines(circuit, label, previous_mse, alpha=50.):
    '''
    The output is: step_reward, previous mse
    '''
    generated_dm = np.asarray(circuit().state())
    _mse = mse(generated_dm, label)*alpha
    #print('step mse= ', _mse)
    if  _mse > previous_mse:
        return 0., _mse
    else:
        if _mse > 1.:
            reward=(1-_mse)*0.5 #lo premio parzialmente perche e migliorato ma la mse e alta
        else:
            reward=(1-_mse) #lo premio a pieno ma meno della final_reward
        return  reward,_mse


def step_reward(circuit, noisy_channels, label, previous_mse, alpha=1.):
    generated_circuit = generate_circuit(circuit, noisy_channels)
    generated_dm = np.asarray(generated_circuit().state())
    _mse = mse(generated_dm, label)
    
    if  _mse > previous_mse:
        return -alpha, _mse
    else:
        return alpha, _mse

def dm_reward(circuit, noisy_channels, label):
    generated_circuit = generate_circuit(circuit, noisy_channels)
    generated_dm = np.asarray(generated_circuit().state())
    return compute_reward(generated_dm, label)

def mse(a,b):
    return np.sqrt(np.abs(((a-b)**2).mean()))

def compute_reward(a,b,alpha=50.,t=1.):
    mse = alpha*np.sqrt(np.abs(((a-b)**2).mean()))
    #print('mse= ',mse)
    if mse > t:
        return 0.
    else:
        return (1.-mse)*3. #occhio che se mse e maggiore di 1 restituisco una reward negativa!

def generate_circuit(circuit, noisy_channels, dep_error=0.05):
    qibo_circuit = Circuit(1, density_matrix=True)
    for i in range(len(circuit)):
        if circuit[i,0]==0:
            qibo_circuit.add(gates.RZ(0, theta=circuit[i,1]*2*np.pi, trainable=False))
        else:
            qibo_circuit.add(gates.RX(0, theta=circuit[i,1]*2*np.pi, trainable=False))
        if noisy_channels[i]==1:
            qibo_circuit.add(gates.DepolarizingChannel((0,), lam=dep_error))
    return qibo_circuit