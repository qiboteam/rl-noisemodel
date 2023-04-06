import numpy as np
from qibo import gates
from qibo.models import Circuit 
from rlnoise.utils import truncated_moments_matching
import copy

def obs_reward(circuit, noisy_channels, label, n_shots=100, reward_func=truncated_moments_matching):
    reward=0.
    generated_circuit = generate_circuit(circuit, noisy_channels)
    observables = np.ndarray((3,2), dtype=float)
    index=0
    for obs in ["Z", "Y", "X"]:
        moments=pauli_probabilities(generated_circuit, obs, n_shots=n_shots)
        observables[index, :]=moments
        index+=1
    for i in range(3):
        reward+=reward_func(m1=observables[i,0], m2=label[i,0], v1=observables[i,1], v2=label[i,1])
    return reward

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

def pauli_probabilities(circuit, observable, n_shots=100, n_rounds=100):
    measured_circuit = copy.deepcopy(circuit)
    add_masurement_gates(measured_circuit, observable=observable)
    register=np.ndarray((n_rounds,), dtype=float)
    moments=np.ndarray((2,), dtype=float)
    for i in range(n_rounds):
        probs=compute_shots(measured_circuit, n_shots=n_shots)
        register[i]=probs[0]-probs[1]
    moments[0]=np.mean(register)
    moments[1]=np.var(register)
    return moments

def add_masurement_gates(circuit, observable):
    if observable=='X' or observable=='Y':
        circuit.add(gates.H(0))
    if observable=='Y':
        circuit.add(gates.SDG(0))
    circuit.add(gates.M(0))
        
def compute_shots(circuit, n_shots):
    shots_register_raw = circuit(nshots=n_shots).frequencies(binary=False)
    shots_register=tuple(int(shots_register_raw[key]) for key in range(2))
    return np.asarray(shots_register, dtype=float)/float(n_shots)