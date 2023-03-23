import sys
from rlnoise.dataset import Dataset, CircuitRepresentation
from rlnoise.policy import CNNFeaturesExtractor
import numpy as np
from rlnoise.envs.gym_env import QuantumCircuit
from stable_baselines3 import PPO, DQN, DDPG
from qibo.noise import DepolarizingError, NoiseModel
from qibo import gates
from rlnoise.rewards.density_matrix_reward import dm_reward_stablebaselines

nqubits = 1
depth = 5
ncirc = 10
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

dataset.set_mode('rep')

circuits=dataset[:]
circuit_env = QuantumCircuit(
    circuits = dataset[:],
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
        filter_shape = (2, nqubits * rep.encoding_dim )
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

labels = dataset.get_dm_labels()
test_sample=0
avg_untrained_rew=0.
# Untrained Agent
for i in range(ncirc):
    obs = circuit_env.reset(i=i)
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = circuit_env.step(action)
    untrained_circ = rep.array_to_circuit(obs[:,:,:-1][0])
    dm_untrained=untrained_circ().state()
    label_dm = labels[i]
    avg_untrained_rew += dm_reward_stablebaselines(noisy_circuit,label_dm)

# Train
model.learn(20000, progress_bar=True) #probably to put inside the for loop

avg_trained_rew=0.
# Trained Agent
for i in range(ncirc):
    obs = circuit_env.reset(i=i)
    #print('Circuit %d representation \n'%(i),rep.array_to_circuit(obs[:,:,:-1][0]).draw()) #test to check that works for more circuits
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = circuit_env.step(action)
    trained_circ = rep.array_to_circuit(obs[:,:,:-1][0])
    dm_trained=trained_circ().state()
    label_dm = labels[i]
    avg_trained_rew += dm_reward_stablebaselines(trained_circ,label_dm)
    if i==test_sample:
        test_dm=dm_trained
        test_circ=trained_circ.copy()
    


print('---- Original Circuit ----\n', circuit.draw(), '\n', circuit_rep)
print(' --> With noise\n', noisy_circuit.draw())
print(labels[test_sample])

print('---- Avg rew Before Training ----\n')
print(avg_untrained_rew/ncirc)

print('---- Avg rew After Training ----\n')
print(avg_trained_rew/ncirc)
print("Test DM", test_dm)
print("Test circ")
print(test_circ.draw())