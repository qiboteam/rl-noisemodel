import numpy as np
from copy import deepcopy
from rlnoise.dataset import CircuitRepresentation
from rlnoise.custom_noise import CustomNoiseModel
from rlnoise.policy import CNNFeaturesExtractor
from rlnoise.rewards.rewards import DensityMatrixReward
from rlnoise.hardware_test import classical_shadows
from rlnoise.utils import RL_NoiseModel
from stable_baselines3 import PPO
from rlnoise.gym_env import QuantumCircuit
import matplotlib.pyplot as plt
from qibo.models import QFT, Circuit
from qibo import gates
from qibo.transpiler.unroller import Unroller, NativeGates

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

noise_model = CustomNoiseModel()
noisy_circuit = noise_model.apply(deepcopy(final_circuit))
print("NOISY CIRCUIT")
print(noisy_circuit.draw())

#IMPLEMENTING A CUSTUM POLICY NETWORK (e.g. increasing dimension of value network) COULD BE AN IDEA
dataset_path= 'src/rlnoise/simulation_phase/3Q_training_new/'
model_path = 'src/rlnoise/saved_models/'
results_filename = f'{dataset_path}train_results'

config_path = 'src/rlnoise/config.json'


#loading benchmark datasets (model can be trained with circuits of different lenghts if passed as list)
circuits_depth=7
nqubits=3
n_circuit_in_dataset=500
dataset_name = 'train_set_D7_3Q_len500.npz'
f = open(dataset_path+dataset_name,"rb")
tmp=np.load(f,allow_pickle=True)
train_set=deepcopy(tmp['train_set'])
train_label=deepcopy(tmp['train_label'])
val_set=deepcopy(tmp['val_set'])
val_label=deepcopy(tmp['val_label'])

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
# clip_range=0.08,
# n_epochs=2
)

agent = model.load("src/rlnoise/simulation_phase/3Q_training_new/3Q_D7_Simulation296000.zip")
rl_noise_model = RL_NoiseModel(agent = agent, circuit_representation = rep)

rl_noisy_circuit = rl_noise_model.apply(deepcopy(final_circuit))
print("RL NOISY CIRCUIT")
print(rl_noisy_circuit.draw())
        
