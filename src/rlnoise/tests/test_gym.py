import sys
from rlnoise.dataset import Dataset, CircuitRepresentation
from rlnoise.policy import CNNFeaturesExtractor
import numpy as np
from rlnoise.envs.gym_env import QuantumCircuit
from stable_baselines3 import PPO, DQN, DDPG
from qibo.noise import DepolarizingError, NoiseModel
from qibo import gates
#from rlnoise.utils import SaveOnBestTrainingRewardCallback

nqubits = 1
depth = 10
ncirc = 1
val_split = 0.2

noise_model = NoiseModel()
lam = 0.2
noise_model.add(DepolarizingError(lam), gates.RZ)
noise_channel = gates.DepolarizingChannel((0,), lam=lam)
primitive_gates = ['RZ', 'RX']
channels = ['DepolarizingChannel']

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

# input circuit
circuit_rep = dataset[0]
dataset.set_mode('circ')
circuit = dataset[0]
dataset.set_mode('noisy_circ')
noisy_circuit = dataset[0]

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

dataset.set_mode('rep')
circuit_env = QuantumCircuit(
    circuits = dataset[0][np.newaxis,:,:],
    noise_channel = noise_channel,
    representation = rep,
    labels = dataset.get_dm_labels(),
    reward_method="dm"
)

policy = "MlpPolicy"
policy_kwargs = dict(
    features_extractor_class = CNNFeaturesExtractor,
    features_extractor_kwargs = dict(
        features_dim = 64,
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
obs = circuit_env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = circuit_env.step(action)
untrained_circ = rep.array_to_circuit(obs[:,:,:-1][0])


#SAVE TRAIN BEST MODEL
#callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

# Train
model.learn(10000, progress_bar=True)

# Trained Agent

obs = circuit_env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = circuit_env.step(action)
trained_circ = rep.array_to_circuit(obs[:,:,:-1][0])

print('---- Original Circuit ----\n', circuit.draw(), '\n', circuit_rep)
print(' --> With noise\n', noisy_circuit.draw())#, '\n', noisy_rep)

print('---- Before Training ----\n', untrained_circ.draw())

print('---- After Training ----\n', trained_circ.draw())