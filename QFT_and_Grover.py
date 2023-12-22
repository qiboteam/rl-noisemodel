import numpy as np
import argparse
from copy import deepcopy
from rlnoise.dataset import CircuitRepresentation
from rlnoise.custom_noise import CustomNoiseModel
from rlnoise.policy import CNNFeaturesExtractor
from rlnoise.rewards import DensityMatrixReward
from rlnoise.metrics import compute_fidelity
from rlnoise.utils import RL_NoiseModel, unroll_circuit
from stable_baselines3 import PPO
from rlnoise.gym_env import QuantumCircuit
from qibo.models import QFT, Circuit
from qibo import gates
from qibo.noise import NoiseModel, DepolarizingError

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.json')
parser.add_argument('--model', type=str, default="simulation_phase/3Q_training_new/3Q_D7_Simulation1.zip")
parser.add_argument('--dataset', type=str, default="simulation_phase/3Q_training_new/train_set_D7_3Q_len500.npz")  
args = parser.parse_args()

def grover():
    """Creates a Grover circuit with 3 qubits.
    The circuit searches for the 11 state, the last qubit is ancillary"""
    circuit = Circuit(3, density_matrix=True)
    circuit.add(gates.H(0))
    circuit.add(gates.H(1))
    circuit.add(gates.X(2))
    circuit.add(gates.H(2))
    circuit.add(gates.TOFFOLI(0, 1, 2))
    circuit.add(gates.H(0))
    circuit.add(gates.H(1))
    circuit.add(gates.X(0))
    circuit.add(gates.X(1))
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.X(0))
    circuit.add(gates.X(1))
    circuit.add(gates.H(0))
    circuit.add(gates.H(1))
    return circuit

circuit_type = "QFT"

if circuit_type == "QFT":
    circuit = QFT(3, with_swaps=False)
if circuit_type == "Grover":
    circuit = grover()

final_circuit = unroll_circuit(circuit)

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

noise = NoiseModel()
noise.add(DepolarizingError(0.10))
RB_noisy_circuit = noise.apply(final_circuit)
        
print("Circuit type", circuit_type)
print("Length", len(final_circuit.queue))
print("No noise", compute_fidelity(noisy_circuit().state(), final_circuit().state()))
print("RL agent", compute_fidelity(noisy_circuit().state(), rl_noisy_circuit().state()))
print("RB noise", compute_fidelity(noisy_circuit().state(), RB_noisy_circuit().state()))

def copy_circ(circ):
    new_circ = Circuit(3, density_matrix=True)
    for gate in circ.queue:
        new_circ.add(gate)
    return new_circ
        

final_circuit2 = copy_circ(final_circuit)
final_circuit2.add(gates.M(0,1,2))
noisy_circuit2 = copy_circ(noisy_circuit)
noisy_circuit2.add(gates.M(0,1,2))
rl_noisy_circuit2 = copy_circ(rl_noisy_circuit)
rl_noisy_circuit2.add(gates.M(0,1,2))
RB_noisy_circuit2 = copy_circ(RB_noisy_circuit)
RB_noisy_circuit2.add(gates.M(0,1,2))

no_noise_shots = final_circuit2.execute(nshots=10000)
noise_shots = noisy_circuit2.execute(nshots=10000)
rl_shots = rl_noisy_circuit2.execute(nshots=10000)
RB_shots = RB_noisy_circuit2.execute(nshots=10000)

print("No noise", no_noise_shots.frequencies())
print("Noise", noise_shots.frequencies())
print("RL", rl_shots.frequencies())
print("RB", RB_shots.frequencies())