import sys
sys.path.append('../rewards/')
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
from rlnoise.rewards.rewards import FrequencyReward,DensityMatrixReward
from rlnoise.dataset import Dataset, CircuitRepresentation
from rlnoise.policy import CNNFeaturesExtractor
import numpy as np
from rlnoise.gym_env import QuantumCircuit
from stable_baselines3 import PPO, DQN, DDPG
from qibo.noise import DepolarizingError, NoiseModel
from qibo import gates
from old.density_matrix_reward import dm_reward_stablebaselines
import qibo
qibo.set_backend('qibojit','numba')

nqubits = 2
depth = 25
ncirc = 2
val_split = 0.2

noise_model = NoiseModel()
lam = 0.2
noise_model.add(DepolarizingError(lam), gates.RZ)
noise_channel = gates.DepolarizingChannel((0,), lam=lam)
primitive_gates = ['RZ', 'RX','CZ']
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
#Setting up the environment
test_sample = np.random.randint(ncirc)
test_sample=0
circuits=dataset[:]
#reward = FrequencyReward()
reward = DensityMatrixReward()
#labels = list(dataset.get_frequencies())
labels=np.array(dataset.get_dm_labels())
circuit_list=[]
#generate list of circuits of different lenght
'''
for i in range(len(dataset[:])):
    if i<20:
        circuit_list.append(dataset[i])
    elif i>20 and i<=50: 
        circuit_list.append(dataset[i][:-2,:])
    elif i>50 and i<=70: 
        circuit_list.append(dataset[i][:-6,:])
    else:
        circuit_list.append(dataset[i][:-10,:])

print('\n\n-----Testing environment on circuits of depths: %d,%d,%d and %d -----'%(circuit_list[10].shape[0],circuit_list[30].shape[0],circuit_list[60].shape[0],circuit_list[90].shape[0]) )
'''
circuit_env = QuantumCircuit(
    circuits = circuits,
    representation = rep,
    labels = labels,
    reward = reward,
    noise_param_space = noise_param_space,

)
#Setting up the policy
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

                        #Testing the environment
for i in range(len(circuits)):                        
    obs = circuit_env.reset(i)
    #print('obs shape: ', obs.shape)
    #print('position: ',int(circuit_env.get_position()))
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        print('action shape:',action.shape)
        obs, rewards, done, info = circuit_env.step(action)
    untrained_rep = obs[:,:,:-1][0]
    #untrained_circ = rep.array_to_circuit(obs[:,:,:][0])
    #print(circuit_env.get_qibo_circuit().draw())
    #dm_untrained = untrained_circ().state()
print('----All done----')
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