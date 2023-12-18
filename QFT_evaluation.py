import numpy as np
import argparse
from copy import deepcopy
from rlnoise.dataset import CircuitRepresentation
from rlnoise.custom_noise import CustomNoiseModel
from rlnoise.policy import CNNFeaturesExtractor
from rlnoise.rewards.rewards import DensityMatrixReward
from rlnoise.hardware_test import classical_shadows
from rlnoise.metrics import compute_fidelity
from rlnoise.utils import RL_NoiseModel
from stable_baselines3 import PPO
from rlnoise.gym_env import QuantumCircuit
import matplotlib.pyplot as plt
from qibo.models import QFT, Circuit
from qibo import gates
from qibo.transpiler.unroller import Unroller, NativeGates

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--dataset', type=str)  
args = parser.parse_args()

def u3_dec(gate):
    # t, p, l = gate.parameters
    params = gate.parameters
    t = params[0]
    p = params[1]
    l = params[2]
    #print("parameters", params)
    decomposition = []
    if l != 0.0:
        decomposition.append(gates.RZ(gate.qubits[0], l))
    decomposition.append(gates.RX(gate.qubits[0], np.pi/2, 0))
    if t != -np.pi:
        decomposition.append(gates.RZ(gate.qubits[0], t + np.pi))
    decomposition.append(gates.RX(gate.qubits[0], np.pi/2, 0))
    if p != -np.pi:
        decomposition.append(gates.RZ(gate.qubits[0], p + np.pi))
    return decomposition


circuit = QFT(3, with_swaps=False)
natives = NativeGates.U3 | NativeGates.CZ
unroller = Unroller(native_gates = natives)

unrolled_circuit = unroller(circuit)
queue = unrolled_circuit.queue
final_circuit = Circuit(3)
for gate in queue:
    if isinstance(gate, gates.CZ):
        final_circuit.add(gate)
    elif isinstance(gate, gates.RZ):
        final_circuit.add(gate)
    elif isinstance(gate, gates.U3):
        decomposed = u3_dec(gate)
        for decomposed_gate in decomposed:
            final_circuit.add(decomposed_gate)

print("TRANSPILED CIRCUIT")
print(final_circuit.draw())

noise_model = CustomNoiseModel(config_file=args.config)
noisy_circuit = noise_model.apply(deepcopy(final_circuit))
print("NOISY CIRCUIT")
print(noisy_circuit.draw())

#IMPLEMENTING A CUSTUM POLICY NETWORK (e.g. increasing dimension of value network) COULD BE AN IDEA


#loading benchmark datasets (model can be trained with circuits of different lenghts if passed as list)

tmp=np.load(args.dataset,allow_pickle=True)
train_set=deepcopy(tmp['train_set'])
train_label=deepcopy(tmp['train_label'])
val_set=deepcopy(tmp['val_set'])
val_label=deepcopy(tmp['val_label'])
n_circuit_in_dataset = train_set.shape[0] + val_set.shape[0]
nqubits = train_set.shape[2]
circuits_depth = train_set.shape[1]

#Setting up training env and policy model
reward = DensityMatrixReward()
rep = CircuitRepresentation()

circuit_env_training = QuantumCircuit(
    circuits = train_set,
    representation = rep,
    labels = train_label,
    reward = reward,
)

policy = "MlpPolicy"
policy_kwargs = dict(
    features_extractor_class = CNNFeaturesExtractor,
    features_extractor_kwargs = dict(
        features_dim = 32,
        filter_shape = (3, 1)
    ),
    net_arch=dict(pi=[32, 32], vf=[32, 32])
)

model= PPO(
policy,
circuit_env_training,
policy_kwargs=policy_kwargs, 
verbose=0,
)
print('MODEL')
agent = model.load(args.model)
rl_noise_model = RL_NoiseModel(agent = agent, circuit_representation = rep)

rl_noisy_circuit = rl_noise_model.apply(deepcopy(final_circuit))
print("RL NOISY CIRCUIT")
print(rl_noisy_circuit.draw())
        
print(compute_fidelity(noisy_circuit().state(), rl_noisy_circuit().state()))