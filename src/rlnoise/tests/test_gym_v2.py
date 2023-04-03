import sys
from rlnoise.rewards.rewards import FrequencyReward,DensityMatrixReward
from rlnoise.dataset import Dataset, CircuitRepresentation
from rlnoise.policy import CNNFeaturesExtractor
import numpy as np
from rlnoise.envs.gym_env_v2 import QuantumCircuit
from stable_baselines3 import PPO, DQN, DDPG
from qibo.noise import DepolarizingError, NoiseModel
from qibo import gates
from rlnoise.rewards.density_matrix_reward import dm_reward_stablebaselines

nqubits = 1
depth = 11
ncirc = 100

noise_model = NoiseModel()
lam = 0.1
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

test_sample = np.random.randint(ncirc)

# input circuit
circuit_rep = dataset[test_sample]
dataset.set_mode('circ')
circuit = dataset[test_sample]
dataset.set_mode('noisy_circ')
noisy_circuit = dataset[test_sample]
#labels = list(dataset.get_frequencies())
labels=np.array(dataset.get_dm_labels())

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
reward = DensityMatrixReward()

circuits=dataset[:]
circuit_env = QuantumCircuit(
    circuits = circuits,
    noise_channel = noise_channel,
    representation = rep,
    labels = labels,
    reward = reward,
    kernel_size=3
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


avg_untrained_rew =0 

# Untrained Agent
for i in range(ncirc):
    obs = circuit_env.reset(i=i)
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = circuit_env.step(action)
    untrained_circ = circuit_env.get_qibo_circuit()
    #print('drawing untrained circuits %d: \n'%(i),untrained_circ.draw())
    dm_untrained=np.array(untrained_circ().state())
    #print('dm untrained label %d: \n'%(i), dm_untrained)
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
    trained_circ = circuit_env.get_qibo_circuit()
    dm_trained=np.array(trained_circ().state())
    label_dm = labels[i]
    avg_trained_rew += dm_reward_stablebaselines(trained_circ,label_dm)
    if i==test_sample:
        test_dm=dm_trained
        test_circ=trained_circ
    


print('---- Original Circuit ----\n', circuit.draw(), '\n', circuit_rep)
print(' --> With noise\n', noisy_circuit.draw())
print(labels[test_sample])

print('---- Avg rew Before Training ----\n')
print(avg_untrained_rew/ncirc)

print('---- After Training ----\n', test_circ.draw())
print(test_dm)
print(avg_trained_rew/ncirc)
