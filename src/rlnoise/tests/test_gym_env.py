import sys
sys.path.append('../')
sys.path.append('../envs/')
from dataset import Dataset
from policy import CNNFeaturesExtractor
import numpy as np
from gym_env import CircuitsGym, QuantumCircuit
from stable_baselines3 import PPO

nqubits = 1
ngates = 10
ncirc = 50
val_split=0.2

# create dataset
dataset = Dataset(
    n_circuits = ncirc,
    n_gates = ngates,
    n_qubits = nqubits,
)

print('Circuits')
for c in dataset.get_circuits():
    print(c.draw())
circuits_repr = dataset.generate_dataset_representation()
dataset.add_noise(noise_params=0.05)
labels = dataset.generate_labels()
#print(labels)


#circuit_env = CircuitsGym(circuits_repr, labels)
circuit_env = QuantumCircuit(circuits_repr, labels, None)

#from stable_baselines3.common.env_checker import check_env
#check_env(circuit_env)

policy_kwargs = dict(
    features_extractor_class=CNNFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=64),
)

model = PPO("MlpPolicy", circuit_env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(4096, progress_bar=True)

"""
circuit_env.reset(verbose=True)
for _ in range(ngates):
    action = np.random.randint(0,2)
    circuit_env.step(action, verbose=True)
"""
