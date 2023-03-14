import sys
sys.path.append('../')
sys.path.append('../envs/')
from dataset import Dataset
from policy import CNNFeaturesExtractor
import numpy as np
from gym_env import CircuitsGym, QuantumCircuit
from stable_baselines3 import PPO, DQN, DDPG
from qibo.noise import DepolarizingError, NoiseModel
from qibo import gates

nqubits = 1
ngates = 5
ncirc = 1
val_split = 0.2

noise_model = NoiseModel()
noise_model.add(DepolarizingError(0.5), gates.RX)
noise_channel = gates.DepolarizingChannel((0,), lam=0.5)

# create dataset
dataset = Dataset(
    n_circuits = ncirc,
    n_gates = ngates,
    n_qubits = nqubits,
    primitive_gates = ['RX', 'RZ'],
    noise_model = noise_model,
    mode = 'rep'
)

print('Circuits')
for c in dataset.get_circuits():
    print(c.draw())

original_rep = dataset[0]
dataset.set_mode('circ')
original_circuit = dataset[0]
original_circuit.add(gates.M(0))
labels = original_circuit(nshots=10000).frequencies()

dataset.set_mode('rep')
circuit_env = QuantumCircuit(
    circuits = dataset[0].reshape(1,ngates,2),
    noise_channel = noise_channel,
    labels = labels,
    reward_f = None
)

policy = "MlpPolicy"
policy_kwargs = dict(
    features_extractor_class = CNNFeaturesExtractor,
    features_extractor_kwargs = dict(features_dim=64),
)

"""
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

# Untrained Agent
obs = circuit_env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = circuit_env.step(action)
c = dataset.repr_to_circuit(obs, noise_channel)
c.add(gates.M(0))
before = c(nshots=10000).frequencies()

# Train
model.learn(81920, progress_bar=True)

# Trained Agent
obs = circuit_env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = circuit_env.step(action)
c = dataset.repr_to_circuit(obs, noise_channel)
c.add(gates.M(0))
after = c(nshots=10000).frequencies()

print('---- Original Circuit ----\n', original_circuit.draw(),'\n', original_rep)
print('---- Noise Model Learnt ----\n', c.draw(), '\n', obs[0])

print('---- Before Training ----')
print(before)

print('---- After Training ----')
print(after)

print('---- Ground Truth ----')
print(labels)
"""
circuit_env.reset(verbose=True)
for _ in range(ngates):
    action = np.random.randint(0,2)
    circuit_env.step(action, verbose=True)
"""
