import numpy as np
from rlnoise.dataset import CircuitRepresentation
from rlnoise.rewards import DensityMatrixReward
from rlnoise.policy import CNNFeaturesExtractor,CustomCallback
from rlnoise.gym_env import QuantumCircuit
from stable_baselines3 import PPO

rep = CircuitRepresentation()

#loading benchmark datasets (model can be trained with circuits of different lenghts if passed as list)
circuits_depth=15
nqubits=3
n_circuit_in_dataset=500
train_size = 300

hardware_results_set = 'src/rlnoise/hardware_test/dm_3Q_IBM/Hanoi_optimized_and_fixed/train_set_Hanoi_3Q.npy'

f = open(hardware_results_set,"rb")
data = np.load(f,allow_pickle=True)
f.close()
training_dm_mit = data[:train_size,3]
training_circ_rep = np.array([CircuitRepresentation().circuit_to_array(circ) for circ in data[:train_size,0]], dtype=object)
training_dm_true = data[:train_size,1]

evaluation_dm_mit = data[train_size:,3]
evaluation_circ_rep = np.array([CircuitRepresentation().circuit_to_array(circ) for circ in data[train_size:,0]], dtype=object)
evaluation_dm_true = data[train_size:,1]
#Setting up training env and policy model

dataset={'train_set': training_circ_rep,
         'train_label': training_dm_mit,
         'val_set': evaluation_circ_rep,
         'val_label': evaluation_dm_mit}

reward = DensityMatrixReward()

circuit_env_training = QuantumCircuit(
    circuits = training_circ_rep,
    representation = rep,
    labels = training_dm_mit,
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
#model=PPO.load(model_path+"rew_each_step_D7_box")

#SINGLE TRAIN AND VALID

callback=CustomCallback(check_freq=5000,evaluation_set=dataset,
                        train_environment=circuit_env_training,
                        trainset_depth=circuits_depth,verbose=True)                                          
model = PPO(
policy,
circuit_env_training,
policy_kwargs=policy_kwargs, 
verbose=0,
clip_range=0.3,
#n_epochs=5,
)

model.learn(300000,progress_bar=True,callback=callback)

