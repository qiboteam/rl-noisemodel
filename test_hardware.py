import numpy as np
import os
from configparser import ConfigParser
import copy
from rlnoise.dataset import CircuitRepresentation
from rlnoise.rewards.rewards import FrequencyReward,DensityMatrixReward
from rlnoise.policy import CNNFeaturesExtractor,CustomCallback
from rlnoise.gym_env import QuantumCircuit
from stable_baselines3 import PPO

rep = CircuitRepresentation()

#loading benchmark datasets (model can be trained with circuits of different lenghts if passed as list)
circuits_depth=5
nqubits=1
n_circuit_in_dataset=500

#benchmark_circ_path=os.getcwd()+'/src/rlnoise/bench_dataset/'
benchmark_dm_train=os.getcwd()+'/src/rlnoise/dataset_hardware/30_05_2023/density_matrices_training2.npy'
benchmark_dm_val=os.getcwd()+'/src/rlnoise/dataset_hardware/30_05_2023/density_matrices_validation2.npy'


f = open(benchmark_dm_train,"rb")
tmp=np.load(f,allow_pickle=True)
training_circ_rep=np.array([copy.deepcopy(rep.circuit_to_array(tmp[i,0])) for i in range(tmp.shape[0])])
training_dm_mit=np.array(copy.deepcopy(tmp[:,3]))
train_dm_true=np.array(copy.deepcopy(tmp[:,1]))
f.close()

f2 = open(benchmark_dm_val,"rb")
tmp2=np.load(f2,allow_pickle=True)
val_circ_rep=np.array([copy.deepcopy(rep.circuit_to_array(tmp2[i,0])) for i in range(tmp2.shape[0])])
val_dm_mit=np.array(copy.deepcopy(tmp2[:,3]))
val_dm_true=np.array(copy.deepcopy(tmp2[:,1]))
f2.close()

print(training_dm_mit.shape, val_dm_mit.shape)
#Setting up training env and policy model

dataset={'train_set': training_circ_rep,
         'train_label': training_dm_mit,
         'val_set': val_circ_rep,
         'val_label': val_dm_mit}

#print(dataset['train_set'].shape,dataset['train_label'].shape,dataset['val_set'].shape,dataset['val_label'].shape)
#print(type(dataset['train_label'][0]))
print(np.array(training_dm_mit)[0])
print(np.sqrt(np.abs(((val_dm_true-val_dm_mit)**2)).mean()).mean())
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
        features_dim = 64,
        filter_shape = (nqubits,1)
    )
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
)

model.learn(500000,progress_bar=True,callback=callback)

