import numpy as np
import os
from configparser import ConfigParser
import copy
from rlnoise.dataset import Dataset, CircuitRepresentation
from qibo import gates
from rlnoise.rewards.rewards import FrequencyReward,DensityMatrixReward
from rlnoise.policy import CNNFeaturesExtractor,CustomCallback
from rlnoise.gym_env import QuantumCircuit
from stable_baselines3 import PPO,DQN,DDPG #not bad
from stable_baselines3 import DQN,A2C,TD3
from rlnoise.custom_noise import CustomNoiseModel
from rlnoise.utils import model_evaluation

params=ConfigParser()
params.read("src/rlnoise/config.ini") 

neg_reward=params.getfloat('gym_env','neg_reward')
pos_reward=params.getfloat('gym_env','pos_reward')
step_r_metric=params.get('gym_env','step_r_metric')
action_penality=params.getfloat('gym_env','action_penality')
action_space_type=params.get('gym_env','action_space')
kernel_size = params.getint('gym_env','kernel_size')
step_reward=params.getboolean('gym_env','step_reward')
#loading benchmark datasets (model can be trained with circuits of different lenghts if passed as list)
circuits_depth=5
nqubits=1
n_circuit_in_dataset=1000
dataset_name="Coherent-on_Std-on"+"_D%d_%dQ_len%d.npz"%(circuits_depth,nqubits,n_circuit_in_dataset)

benchmark_circ_path=os.getcwd()+'/src/rlnoise/bench_dataset/'
model_path=os.getcwd()+'/src/rlnoise/saved_models/'
bench_results_path=os.getcwd()+'/src/rlnoise/bench_results'

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
    action_penality=action_penality,
    action_space_type=action_space_type,
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
Rew_Mae_TraceD_untrained=[]
Rew_Mae_TraceD_trained=[]

#model=PPO.load(model_path+"rew_each_step_D7_box")

                                                #SINGLE TRAIN AND VALID

callback=CustomCallback(check_freq=5000,evaluation_set=tmp,train_environment=circuit_env_training,trainset_depth=circuits_depth,verbose=1)                                          
model = PPO(
policy,
circuit_env_training,
policy_kwargs=policy_kwargs, 
verbose=0,
n_epochs=20
)

model.learn(500000,progress_bar=True,callback=callback)

f.close()
'''
                                            #TEST A SAVED MODEL ON DIFFERENT DEPTHS
   

model1= PPO(
policy,
circuit_env_training,
policy_kwargs=policy_kwargs, 
verbose=0,
)
model=PPO.load(model_path+"/best_model_Q3_D7154000")

nqubits=3
n_circuit_in_dataset=100
depth_list=[7,10,15,20,25,30,35,40]
result_filename='Dep-Term_CZ_3Q_154k_1'
for d in depth_list:
    dataset_name='3Q_CoherentOnly'+'_D%d_%dQ_len%d.npz'%(d,nqubits,n_circuit_in_dataset)
    f = open(benchmark_circ_path+dataset_name,"rb")
    tmp=np.load(f,allow_pickle=True)
    val_set=tmp['val_set']
    val_label=tmp['val_label']
    f.close()
    val_avg_rew_untrained,mae_untrained,trace_dist_untrain=(model_evaluation(val_set,val_label,circuit_env_training,model1))
    val_avg_rew_trained,mae_trained,trace_dist_train=(model_evaluation(val_set,val_label,circuit_env_training,model))
    Rew_Mae_TraceD_trained.append([val_avg_rew_trained,mae_trained,trace_dist_train])
    Rew_Mae_TraceD_untrained.append( [val_avg_rew_untrained,mae_untrained,trace_dist_untrain])

Rew_Mae_TraceD_trained=np.array(Rew_Mae_TraceD_trained)
Rew_Mae_TraceD_untrained=np.array(Rew_Mae_TraceD_untrained)
f = open(bench_results_path+result_filename+str(depth_list),"wb")
np.savez(f,trained=Rew_Mae_TraceD_trained,untrained=Rew_Mae_TraceD_untrained)
f.close()
'''


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
#f.close()
