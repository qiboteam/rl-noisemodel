from qibo import gates
from qibo.models import Circuit
import numpy as np

def dm_reward(circuit, noisy_channels, label):
    generated_circuit = generate_circuit(circuit, noisy_channels)
    generated_dm = np.asarray(generated_circuit().state())
    return compute_reward(generated_dm, label)

def compute_reward(a,b,alpha=100,t=1):
    mse = alpha*np.sqrt(np.abs(((a-b)**2).mean()))
    if mse > t:
        return 0
    else:
        return 1-mse

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