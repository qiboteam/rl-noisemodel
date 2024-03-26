import copy
import numpy as np
import argparse
from pathlib import Path
from rlnoise.dataset import CircuitRepresentation
from rlnoise.rewards import DensityMatrixReward
from rlnoise.policy import CNNFeaturesExtractor,CustomCallback
from rlnoise.gym_env import QuantumCircuit
from stable_baselines3 import PPO
from rlnoise.custom_noise import CustomNoiseModel

ncirc = 500
current_path = Path(__file__).parent
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=f"{current_path}/src/rlnoise/simulation_phase/1Q_training_new/config.json")
parser.add_argument('--dataset', type=str, default=f"{current_path}/src/rlnoise/simulation_phase/1Q_training_new/train_set_D7_1Q_len500.npz")
parser.add_argument('--output', type=str, default=f"{current_path}/src/rlnoise/simulation_phase/1Q/{ncirc}_circ")
args = parser.parse_args()

#IMPLEMENTING A CUSTUM POLICY NETWORK (e.g. increasing dimension of value network) COULD BE AN IDEA
results_filename = f'{args.output}/train_results_mse_len{ncirc}'

#loading benchmark datasets (model can be trained with circuits of different lenghts if passed as list)
tmp = np.load(args.dataset, allow_pickle=True)
train_set = copy.deepcopy(tmp['train_set'][:ncirc])
train_label = copy.deepcopy(tmp['train_label'][:ncirc])
val_set = copy.deepcopy(tmp['val_set'][:ncirc])
val_label = copy.deepcopy(tmp['val_label'][:ncirc])

#Custom val set
# val_set_tmp = np.load("simulation_phase/3Q_non_clifford/non_clifford_set.npz", allow_pickle=True)
# val_set = copy.deepcopy(val_set_tmp['val_circ'])
# val_label = copy.deepcopy(val_set_tmp['val_label'])

n_circuit_in_dataset = train_set.shape[0] + val_set.shape[0]
nqubits = train_set[0].shape[1]
print(f"nqubits: {nqubits}")
#Setting up training env and policy model

noise_model = CustomNoiseModel(args.config)
reward = DensityMatrixReward()
rep = CircuitRepresentation(args.config)

circuit_env_training = QuantumCircuit(
    circuits = train_set,
    representation = rep,
    labels = train_label,
    reward = reward,
)

policy = "MlpPolicy"
policy_kwargs = dict(
    #activation_fn = torch.nn.Sigmoid,
    features_extractor_class = CNNFeaturesExtractor,
    features_extractor_kwargs = dict(
        features_dim = 32,
        filter_shape = (nqubits,1)
    ),
    net_arch=dict(pi=[32, 32], vf=[32, 32])
)

model= PPO(
policy,
circuit_env_training,
policy_kwargs=policy_kwargs,
verbose=0,
n_steps=256,
)
#                             #STANDARD TRAINING

callback=CustomCallback(check_freq=2500,
                        dataset=tmp,
                        train_environment=circuit_env_training,
                        verbose=True,
                        result_filename=results_filename,
                        )                                          

model.learn(total_timesteps=200000, progress_bar=True, callback=callback)


