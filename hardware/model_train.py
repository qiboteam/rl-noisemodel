import numpy as np
from rlnoise.dataset import CircuitRepresentation
from rlnoise.rewards import DensityMatrixReward
from rlnoise.policy import CNNFeaturesExtractor,CustomCallback
from rlnoise.gym_env import QuantumCircuit
from stable_baselines3 import PPO

config = "src/rlnoise/hardware_test/dm_1Q/config.json"
output_folder = "src/rlnoise/hardware_test/trained_models"

rep = CircuitRepresentation(config)

#loading benchmark datasets (model can be trained with circuits of different lenghts if passed as list)
circuits_depth=11
nqubits=1
train_size = 80
# src/rlnoise/hardware_test/dm_1Q/100_circ_set_result_qubit2_NEW.npy
# src/rlnoise/hardware_test/dm_1Q/100_circ_set_result_new.npy
hardware_results_set = 'src/rlnoise/hardware_test/dm_1Q/100_circ_set_result_new.npy'

with open(hardware_results_set,"rb") as f:
    data = np.load(f,allow_pickle=True)

training_dm = data[:train_size,2]
training_circ_rep = np.array([CircuitRepresentation(config).circuit_to_array(circ) for circ in data[:train_size,0]], dtype=object)
training_dm_true = data[:train_size,1]

evaluation_dm = data[train_size:,2]
evaluation_circ_rep = np.array([CircuitRepresentation(config).circuit_to_array(circ) for circ in data[train_size:,0]], dtype=object)
evaluation_dm_true = data[train_size:,1]
#Setting up training env and policy model

dataset={'train_set': training_circ_rep,
         'train_label': training_dm,
         'val_set': evaluation_circ_rep,
         'val_label': evaluation_dm}

reward = DensityMatrixReward()

circuit_env_training = QuantumCircuit(
    circuits = training_circ_rep,
    representation = rep,
    labels = training_dm,
    reward = reward
)
policy = "MlpPolicy"
policy_kwargs = dict(
    features_extractor_class = CNNFeaturesExtractor,
    features_extractor_kwargs = dict(
        features_dim = 32,
        filter_shape = (nqubits,3),
    ),
        net_arch=dict(pi=[32, 32], vf=[32, 32])
)

callback=CustomCallback(check_freq=2500,dataset=dataset,
                        train_environment=circuit_env_training,
                        verbose=True, result_filename="1Q_hardw_qubit2_forRB",
                        config_path=config, out_folder=output_folder)                                          
model = PPO(
policy,
circuit_env_training,
policy_kwargs=policy_kwargs, 
verbose=0,
clip_range=0.12,
# n_epochs=5,
# n_steps=64
)

model.learn(250000,progress_bar=True,callback=callback)

