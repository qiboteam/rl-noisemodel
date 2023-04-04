import sys
sys.path.append('../rewards/')
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
from rlnoise.rewards.rewards import FrequencyReward,DensityMatrixReward
from rlnoise.dataset import Dataset, CircuitRepresentation
from rlnoise.policy import CNNFeaturesExtractor
import numpy as np
from rlnoise.envs.gym_env import QuantumCircuit
from stable_baselines3 import PPO, DQN, DDPG
from qibo.noise import DepolarizingError, NoiseModel
from qibo import gates
from rlnoise.rewards.density_matrix_reward import dm_reward_stablebaselines
import qibo
qibo.set_backend('qibojit','numba')

nqubits = 1
depth = 10
ncirc = 100
val_split = 0.2

noise_model = NoiseModel()
lam = 0.2
noise_model.add(DepolarizingError(lam), gates.RZ)
noise_channel = gates.DepolarizingChannel((0,), lam=lam)
primitive_gates = ['RZ', 'RX']
channels = ['DepolarizingChannel']
noise_param_space = { 'range': (0,0.2), 'n_steps': 100 } 

rep = CircuitRepresentation(
    primitive_gates = primitive_gates,
    noise_channels = channels,
    shape = '2d'
)

# create dataset
dataset = Dataset(
    n_circuits = ncirc,
    n_gates = depth,
    n_qubits = nqubits,
    representation = rep,
    clifford = True,
    noise_model = noise_model,
    mode = 'rep'
)

test_sample = np.random.randint(ncirc)

# input circuit
circuit_rep = dataset[test_sample]
dataset.set_mode('circ')
circuit = dataset[test_sample]
dataset.set_mode('noisy_circ')
noisy_circuit = dataset[test_sample]
noisy_rep = dataset.noisy_circ_rep[test_sample]
labels = list(dataset.get_frequencies())
#labels=np.array(dataset.get_dm_labels())

def test_representation():
    print('> Noiseless Circuit:\n', circuit.draw())
    array = rep.circuit_to_array(circuit)
    print(' --> Representation:\n', array)
    print(' --> Circuit Rebuilt:\n', rep.array_to_circuit(array).draw())
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('> Noisy Circuit:\n', noisy_circuit.draw())
    array = rep.circuit_to_array(noisy_circuit)
    print(array)
    print(' --> Circuit Rebuilt:\n', rep.array_to_circuit(array).draw())

#test_representation()

reward = FrequencyReward()
#reward = DensityMatrixReward()
dataset.set_mode('rep')
circuit_env = QuantumCircuit(
    circuits = dataset.circ_rep,
    representation = rep,
    labels = labels,
    reward = reward,
    noise_param_space = noise_param_space
)

policy = "MlpPolicy"
policy_kwargs = dict(
    features_extractor_class = CNNFeaturesExtractor,
    features_extractor_kwargs = dict(
        features_dim = 32,
        filter_shape = (4, nqubits * rep.encoding_dim )
    )
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
obs = circuit_env.reset(i=test_sample)
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = circuit_env.step(action)
untrained_rep = obs[:,:,:-1][0]
untrained_circ = rep.array_to_circuit(obs[:,:,:-1][0])
dm_untrained = untrained_circ().state()

# Train
model.learn(50000, progress_bar=True)

# Trained Agent
obs = circuit_env.reset(i=test_sample)
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = circuit_env.step(action)
trained_rep = obs[:,:,:-1][0]
trained_circ = rep.array_to_circuit(obs[:,:,:-1][0])
dm_trained=trained_circ().state()

labels = dataset.get_dm_labels()
label_dm = labels[test_sample]


print('---- Original Circuit ----\n', circuit.draw())
print(' --> With noise\n', noisy_circuit.draw(), '\n', noisy_rep)
print(label_dm)
print(dm_reward_stablebaselines(noisy_circuit,label_dm))

print('---- Before Training ----\n', untrained_circ.draw(), '\n', untrained_rep)
print(dm_untrained)
print(dm_reward_stablebaselines(untrained_circ,label_dm))

print('---- After Training ----\n', trained_circ.draw(), '\n', trained_rep)
print(dm_trained)
print(dm_reward_stablebaselines(trained_circ,label_dm))
