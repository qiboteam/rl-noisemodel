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

current_path = Path(__file__).parent
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=f"{current_path}/src/rlnoise/config_hardware.json")
parser.add_argument('--dataset', type=str, default=f"{current_path}/src/rlnoise/hardware_test/dm_1Q/100_circ_set_result_qubit0.npy")
parser.add_argument('--output', type=str, default=f"{current_path}/src/rlnoise/model_folder")
args = parser.parse_args()

results_filename = f'{args.output}/result_hardware_q0'

tmp = np.load(args.dataset, allow_pickle=True)
# 80 for training, 20 for validation
train_set = copy.deepcopy(tmp[0:80, 0])
val_set = copy.deepcopy(tmp[80:, 0])
# 2 for unmitigated, 1 for mitigated
train_label = copy.deepcopy(tmp[0:80, 0])
val_label = copy.deepcopy(tmp[80:, 0])

rep = CircuitRepresentation(args.config)

train_set = np.array([CircuitRepresentation().circuit_to_array(circ) for circ in train_set])
val_set = np.array([CircuitRepresentation().circuit_to_array(circ) for circ in val_set])

n_circuit_in_dataset = train_set.shape[0] + val_set.shape[0]
nqubits = train_set[0].shape[1]
print(f"nqubits: {nqubits}")

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
n_steps=512,
device="cuda"
)
#                             #STANDARD TRAINING

callback=CustomCallback(check_freq=15000,
                        dataset=tmp,
                        train_environment=circuit_env_training,
                        verbose=True,
                        result_filename=results_filename,
                        )                                          
model.learn(total_timesteps=50000, progress_bar=True, callback=callback)
