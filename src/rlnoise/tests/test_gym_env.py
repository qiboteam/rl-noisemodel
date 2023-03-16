import sys
sys.path.append('../')
sys.path.append('../envs/')
from dataset import Dataset, CircuitRepresentation
from policy import CNNFeaturesExtractor
import numpy as np
from gym_env import CircuitsGym, QuantumCircuit
from stable_baselines3 import PPO, DQN, DDPG
from qibo.noise import DepolarizingError, NoiseModel
from qibo import gates

nqubits = 3
ngates = 10
ncirc = 1
val_split = 0.2

noise_model = NoiseModel()
lam = 0.5
noise_model.add(DepolarizingError(lam), gates.RX)
noise_channel = gates.DepolarizingChannel((0,), lam=lam)
gate2index = {'RX':1, 'RZ':0}
channel2index = {'DepolarizingChannel': 0}
index2gate = {v:getattr(gates, k) for k,v in gate2index.items()}

rep = CircuitRepresentation(
    gates_map = gate2index,
    noise_channels_map = channel2index,
    shape = '2d'
)

# create dataset
dataset = Dataset(
    n_circuits = ncirc,
    n_gates = ngates,
    n_qubits = nqubits,
    clifford = False,
    primitive_gates = gate2index,
    noise_model = noise_model,
    mode = 'rep'
)

# input circuit
circuit_rep = dataset[0]
dataset.set_mode('circ')
circuit = dataset[0]
dataset.set_mode('noisy_circ')
noisy_circuit = dataset[0]
#noisy_rep = dataset.circuit_to_rep(noisy_circuit)

crep = rep.circuit_to_array(circuit)

circuit.add(gates.M(0))
noisy_circuit.add(gates.M(0))
labels = noisy_circuit(nshots=10000).frequencies()

dataset.set_mode('rep')
circuit_env = QuantumCircuit(
    circuits = dataset[0].reshape(1,ngates,2),
    noise_channel = noise_channel,
    index2gate = index2gate,
    labels = labels,
    reward_f = None
)

policy = "MlpPolicy"
policy_kwargs = dict(
    features_extractor_class = CNNFeaturesExtractor,
    features_extractor_kwargs = dict(features_dim=64, filter_shape=(4,4)),
)


model = PPO(
    policy,
    circuit_env,
    policy_kwargs=policy_kwargs,
    verbose=1,
)
"""
model = DQN(
    policy,
    circuit_env,
    policy_kwargs = policy_kwargs,
    verbose = 1,
    learning_starts = 1000, # default 50000
    #exploration_final_eps = 0.01, # default 0.05
    #learning_rate = 1e-5 # default 1e-4
)
"""

# Untrained Agent
obs = circuit_env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = circuit_env.step(action)
untrained_rep = obs
untrained_circ = dataset.rep_to_circuit(obs, noise_channel)
untrained_circ.add(gates.M(0))
untrained_pred = untrained_circ(nshots=10000).frequencies()

# Train
model.learn(18192, progress_bar=True)

# Trained Agent
obs = circuit_env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = circuit_env.step(action)
trained_rep = obs
trained_circ = dataset.rep_to_circuit(obs, noise_channel)
trained_circ.add(gates.M(0))
trained_pred = untrained_circ(nshots=10000).frequencies()

print('---- Original Circuit ----\n', circuit.draw(), '\n', circuit_rep)
print(' --> With noise\n', noisy_circuit.draw())#, '\n', noisy_rep)
print(' --> Frequencies\n', labels)

print('---- Before Training ----\n', untrained_circ.draw(), '\n', untrained_rep)
print(' --> Frequencies\n', untrained_pred)

print('---- After Training ----\n', trained_circ.draw(), '\n', trained_rep)
print(' --> Frequencies\n', trained_pred)

