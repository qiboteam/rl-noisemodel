import os
import json
import copy
import numpy as np
from pathlib import Path
from rlnoise.dataset import CircuitRepresentation
from rlnoise.rewards.rewards import FrequencyReward,DensityMatrixReward
from rlnoise.policy import CNNFeaturesExtractor,CustomCallback
from rlnoise.gym_env import QuantumCircuit
from stable_baselines3 import PPO
from rlnoise.custom_noise import CustomNoiseModel
from rlnoise.utils import model_evaluation
benchmark_circ_path=os.getcwd()+'/src/rlnoise/bench_dataset/'
model_path=os.getcwd()+'/src/rlnoise/saved_models/'
bench_results_path=os.getcwd()+'/src/rlnoise/bench_results'
config_path=str(Path().parent.absolute())+'/src/rlnoise/config.json'

with open(config_path) as f:
    config = json.load(f)

gym_env_params = config['gym_env']
kernel_size = gym_env_params['kernel_size']
step_reward = gym_env_params['step_reward']
step_r_metric = gym_env_params['step_r_metric']
neg_reward = gym_env_params['neg_reward']
pos_reward = gym_env_params['pos_reward']
action_penalty = gym_env_params['action_penalty']
action_space = gym_env_params['action_space']

#loading benchmark datasets (model can be trained with circuits of different lenghts if passed as list)
circuits_depth=5
nqubits=1
n_circuit_in_dataset=1000
dataset_name="Coherent-on_Std-on"+"_D%d_%dQ_len%d.npz"%(circuits_depth,nqubits,n_circuit_in_dataset)

f = open(benchmark_circ_path+dataset_name,"rb")
tmp=np.load(f,allow_pickle=True)
train_set=copy.deepcopy(tmp['train_set'])
train_label=copy.deepcopy(tmp['train_label'])
val_set=copy.deepcopy(tmp['val_set'])
val_label=copy.deepcopy(tmp['val_label'])

#Setting up training env and policy model

noise_model = CustomNoiseModel()
reward = DensityMatrixReward()

rep = CircuitRepresentation()
circuit_env_training = QuantumCircuit(
    circuits = train_set,
    representation = rep,
    labels = train_label,
    reward = reward,
    neg_reward=neg_reward,
    pos_reward=pos_reward,
    step_r_metric=step_r_metric,
    action_penality=action_penalty,
    action_space_type=action_space,
    kernel_size = kernel_size,
    step_reward=step_reward
)
policy = "MlpPolicy"
policy_kwargs = dict(
    features_extractor_class = CNNFeaturesExtractor,
    features_extractor_kwargs = dict(
        features_dim = 32,
        filter_shape = (nqubits,1)
    )
)
#model=PPO.load(model_path+"rew_each_step_D7_box")

                                                #SINGLE TRAIN AND VALID
'''
callback=CustomCallback(check_freq=5000,verbose=True,evaluation_set=tmp,train_environment=circuit_env_training,trainset_depth=circuits_depth)                                          
model = PPO(
policy,
circuit_env_training,
policy_kwargs=policy_kwargs, 
verbose=0,
)

model.learn(1000000,progress_bar=True,callback=callback)

f.close()
'''
                                            #TEST A SAVED MODEL ON DIFFERENT DEPTHS
   
results_list_untrained=[]
results_list_trained=[]
model1= PPO(
policy,
circuit_env_training,
policy_kwargs=policy_kwargs, 
verbose=0,
)
model=PPO.load(model_path+"/1Q_AllNoises_mseReward735000")

nqubits=1
n_circuit_in_dataset=1000
depth_list=[5,7,10,15,30]
result_filename='AllNoise_len1000_1Msteps'
for d in depth_list:
    dataset_name='Coherent-on_Std-on'+'_D%d_%dQ_len%d.npz'%(d,nqubits,n_circuit_in_dataset)
    f = open(benchmark_circ_path+dataset_name,"rb")
    tmp=np.load(f,allow_pickle=True)
    val_set=tmp['val_set']
    val_label=tmp['val_label']
    f.close()
    results_untrained_model = (model_evaluation(val_set,val_label,circuit_env_training,model1))
    results_trained_model = (model_evaluation(val_set,val_label,circuit_env_training,model))
    results_list_trained.append(results_trained_model)
    results_list_untrained.append(results_untrained_model)

results_list_trained=np.array(results_list_trained)
results_list_untrained=np.array(results_list_untrained)
f = open(bench_results_path+result_filename+str(depth_list),"wb")
np.savez(f,trained=results_list_trained,untrained=results_list_untrained)
f.close()



                        #TRAIN & TEST ON DATASET W SAME PARAMS BUT DIFFERENT SIZE(n_circ)
'''
circuits_depth=7                       
n_circ=[10,50,200,400]
f = open(benchmark_circ_path+"/depth_%dDep-Term_CZ_3Q_1000.npz"%(circuits_depth),"rb")
tmp=np.load(f,allow_pickle=True)
val_set=copy.deepcopy(tmp['val_set'])
val_label=copy.deepcopy(tmp['val_label'])
train_set=copy.deepcopy(tmp['train_set'])
train_label=copy.deepcopy(tmp['train_label'])
for data_size in n_circ:

    circuit_env_training = QuantumCircuit(
    circuits = train_set[:data_size],
    representation = rep,
    labels = train_label[:data_size],
    reward = reward,
    kernel_size=kernel_size
    )
    callback=CustomCallback(check_freq=2000,evaluation_set=tmp,train_environment=circuit_env_training,trainset_depth=circuits_depth,test_on_data_size=data_size)                                          
    model = PPO(
    policy,
    circuit_env_training,
    policy_kwargs=policy_kwargs, 
    verbose=0,
    )
    model.learn(100000,progress_bar=True, callback=callback)
f.close()
'''

