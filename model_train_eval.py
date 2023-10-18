import os
import copy
import numpy as np
from pathlib import Path
from rlnoise.dataset import CircuitRepresentation
from rlnoise.rewards.rewards import FrequencyReward,DensityMatrixReward
from rlnoise.policy import CNNFeaturesExtractor,CustomCallback
from rlnoise.gym_env import QuantumCircuit
from stable_baselines3 import PPO
from rlnoise.custom_noise import CustomNoiseModel
from rlnoise.utils import model_evaluation, RB_evaluation
#IMPLEMENTING A CUSTUM POLICY NETWORK (e.g. increasing dimension of value network) COULD BE AN IDEA
benchmark_circ_path= 'src/rlnoise/simulation_phase/1Q_training/'
model_path = 'src/rlnoise/saved_models/'
bench_results_path = 'src/rlnoise/bench_results/'
config_path = 'src/rlnoise/config.json'


#loading benchmark datasets (model can be trained with circuits of different lenghts if passed as list)
circuits_depth=15
nqubits=1
n_circuit_in_dataset=500
dataset_name="train_set"+"_D%d_%dQ_len%d.npz"%(circuits_depth,nqubits,n_circuit_in_dataset)

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
# [print(rep.rep_to_circuit(val_set[i]).draw()) for i in range(len(val_set))]
# [print(val_set[i]) for i in range(len(val_set))]
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
    )
)

model= PPO(
policy,
circuit_env_training,
policy_kwargs=policy_kwargs, 
verbose=0,
clip_range=0.3,
n_epochs=5
)

callback=CustomCallback(check_freq=2000,evaluation_set=tmp,
                        train_environment=circuit_env_training,
                        trainset_depth=circuits_depth, verbose=True)                                          

model.learn(600000,progress_bar=True, callback=callback)

# TESTING A PREVIOUSLY TRAINED MODEL ON DIFFERENT DEPTHS AND COMPARING WITH RB AND WITH UNTRAINED MODEL

# results_list_untrained=[]
# results_list_trained = []
# result_RB_list = []
# model_trained=PPO.load(model_path+"/1Q_D10_AllNoises_mseReward_705000")

# nqubits=1
# n_circuit_in_dataset=50
# depth_list=np.arange(3,31,3)
# result_filename='AllNoise_len50_1Msteps_1Q.npz'
# for d in depth_list:
#     dataset_name='benchmark'+'_D%d_%dQ_len%d.npz'%(d,nqubits,n_circuit_in_dataset)
#     f = open(benchmark_circ_path+dataset_name,"rb")
#     tmp=np.load(f,allow_pickle=True)
#     val_set=tmp['clean_rep']
#     val_label=tmp['label']
#     f.close()
#     #results_untrained_model = (model_evaluation(val_set,val_label,circuit_env_training,model))
#     results_trained_model = model_evaluation(val_set,val_label,model_trained,reward=reward,representation=rep)
#     results_RB = RB_evaluation(lambda_RB=0.07,circ_representation=val_set,target_label=val_label)
#     results_list_trained.append(results_trained_model)
#     result_RB_list.append(results_RB)
#     #results_list_untrained.append(results_untrained_model)
# model_results = np.array(results_list_trained)
# rb_results = np.array(result_RB_list)
# #results_list_untrained=np.array(results_list_untrained)

# f = open(bench_results_path+result_filename,"wb")
# np.savez(f,trained=model_results,RB=rb_results)
# f.close()


          #  TRAINING ON DIFFERENT DATASET SIZE (Evaluating the best dataset size for overfitting)

# circuits_depth=15                    
# n_circ=[10,50,200,400]
# f = open(benchmark_circ_path+"/depth_%dDep-Term_CZ_3Q_1000.npz"%(circuits_depth),"rb")
# tmp=np.load(f,allow_pickle=True)
# val_set=copy.deepcopy(tmp['val_set'])
# val_label=copy.deepcopy(tmp['val_label'])
# train_set=copy.deepcopy(tmp['train_set'])
# train_label=copy.deepcopy(tmp['train_label'])
# for data_size in n_circ:

#     circuit_env_training = QuantumCircuit(
#     circuits = train_set[:data_size],
#     representation = rep,
#     labels = train_label[:data_size],
#     reward = reward,
#     kernel_size=kernel_size
#     )
#     callback=CustomCallback(check_freq=2000,evaluation_set=tmp,train_environment=circuit_env_training,trainset_depth=circuits_depth,test_on_data_size=data_size)                                          
#     model = PPO(
#     policy,
#     circuit_env_training,
#     policy_kwargs=policy_kwargs, 
#     verbose=0,
#     )
#     model.learn(100000,progress_bar=True, callback=callback)
# f.close()


