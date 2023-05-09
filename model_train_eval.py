import numpy as np
import os
import copy
from rlnoise.datasetv2 import Dataset, CircuitRepresentation
from qibo import gates
from rlnoise.rewards.rewards import FrequencyReward,DensityMatrixReward
from rlnoise.policy import CNNFeaturesExtractor,CustomCallback
from rlnoise.gym_env import QuantumCircuit
from stable_baselines3 import PPO,DQN,DDPG #not bad
from stable_baselines3 import DQN,A2C,TD3
from rlnoise.utils import model_evaluation
from rlnoise.CustomNoise import CustomNoiseModel
from rlnoise.MlpPolicy import MlPFeaturesExtractor

#loading benchmark datasets (model can be trained with circuits of different lenghts if passed as list)
circuits_depth=7

benchmark_circ_path=os.getcwd()+'/src/rlnoise/bench_dataset'
model_path=os.getcwd()+'/src/rlnoise/saved_models/'
bench_results_path=os.getcwd()+'/src/rlnoise/bench_results'

f = open(benchmark_circ_path+"/depth_%dDep-Term_CZ_3Q_1000.npz"%(circuits_depth),"rb")
tmp=np.load(f,allow_pickle=True)
train_set=copy.deepcopy(tmp['train_set'])
train_label=copy.deepcopy(tmp['train_label'])
val_set=copy.deepcopy(tmp['val_set'])
val_label=copy.deepcopy(tmp['val_label'])


#Setting up training env and policy model
nqubits=3
noise_model = CustomNoiseModel()

reward = DensityMatrixReward()
kernel_size=3

rep = CircuitRepresentation(
    primitive_gates = noise_model.primitive_gates,
    noise_channels = noise_model.channels,
    shape = '3d',
    coherent_noise=False
)
circuit_env_training = QuantumCircuit(
    circuits = train_set,
    representation = rep,
    labels = train_label,
    reward = reward,
    kernel_size=kernel_size,
    step_reward=False
)
policy = "MlpPolicy"
policy_kwargs = dict(
    features_extractor_class = CNNFeaturesExtractor,
    features_extractor_kwargs = dict(
        features_dim = 64,
        filter_shape = (nqubits,1)
    )
)
Rew_Mae_TraceD_untrained=[]
Rew_Mae_TraceD_trained=[]

#model=PPO.load(model_path+"rew_each_step_D7_box")

                                                #SINGLE TRAIN AND VALID

callback=CustomCallback(check_freq=5000,evaluation_set=tmp,train_environment=circuit_env_training,trainset_depth=circuits_depth,save_best=True)                                          
model = PPO(
policy,
circuit_env_training,
policy_kwargs=policy_kwargs, 
verbose=0,
)


#val_avg_rew_untrained,mae_untrained,trace_dist_untr=(model_evaluation(val_set,val_label,circuit_env_training,model))
model.learn(500000,progress_bar=True,callback=callback)
#val_avg_rew_trained,mae_trained,trace_dist_train=(model_evaluation(val_set,val_label,circuit_env_training,model))
#print('avg reward from untrained model: %f\n'%(val_avg_rew_untrained),'avg reward from trained model: %f \n'%(val_avg_rew_trained))
#print('avg MAE from untrained model: %f\n'%(mae_untrained*10),'avg MAE from trained model: %f \n'%(mae_trained*10))
#print('avg Trace Distance from untrained model: %f\n'%(trace_dist_untr),'avg Trace Distance from trained model: %f \n'%(trace_dist_train))
#if trace_dist_train ==0:
#    model.save(model_path+"D5_K3_1Q_Dep_Therm_30k")
#model.save(model_path+"D7_K3_3Q_Dep0.005_Therm0.07_80k")
f.close()
                                        #TRAIN & TEST ON SAME DEPTH BUT DIFFERENT TIMESTEPS
'''
total_timesteps=[2000,4000,6000,8000,10000,15000,20000,30000,50000,80000,130000,200000]

model = PPO(
policy,
circuit_env_training,
policy_kwargs=policy_kwargs, 
verbose=0,
)

for time in total_timesteps:
    model = PPO(
    policy,
    circuit_env_training,
    policy_kwargs=policy_kwargs, 
    verbose=0,
    )
    val_avg_rew_untrained,mae_untrained,trace_dist_untr=(model_evaluation(val_set,val_label,circuit_env_training,model))
    Rew_Mae_TraceD_untrained.append([val_avg_rew_untrained,mae_untrained,trace_dist_untr])

    model.learn(time, progress_bar=True) 

    val_avg_rew_trained,mae_trained,trace_dist_train=(model_evaluation(val_set,val_label,circuit_env_training,model))
    if trace_dist_train <0.0001:
        model.save(model_path+'/Dep-Term_CZ_3Q_BestMod'+str(time))
    Rew_Mae_TraceD_trained.append([val_avg_rew_trained,mae_trained,trace_dist_train])
    del model
Rew_Mae_TraceD_untrained=np.array(Rew_Mae_TraceD_untrained)
Rew_Mae_TraceD_trained=np.array(Rew_Mae_TraceD_trained)

f = open(bench_results_path+"/Dep-Term_CZ_3Q"+str(total_timesteps),"wb")
np.savez(f,untrained=Rew_Mae_TraceD_untrained,trained=Rew_Mae_TraceD_trained)
f.close()
'''
                                            #TRAIN AND TEST ON DIFFERENT DEPTHS
   
'''
model1= PPO(
policy,
circuit_env_training,
policy_kwargs=policy_kwargs, 
verbose=0,
)
model=PPO.load(model_path+"/best_model_Q3_D7154000")
depth_list=[7,10,15,20,25,30,35,40]
for d in depth_list:
    f = open(benchmark_circ_path+"/depth_%dDep-Term_CZ_3Q.npz"%(d),"rb")
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
f = open(bench_results_path+"/Dep-Term_CZ_3Q_154k_1"+str(depth_list),"wb")
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


#print('avg reward from untrained model: %f\n'%(val_avg_rew_untrained),'avg reward from trained model: %f \n'%(val_avg_rew_trained))
#print('avg MAE from untrained model: %f\n'%(mea_untrained*10),'avg MAE from trained model: %f \n'%(mae_trained*10))
#print('avg Trace Distance from untrained model: %f\n'%(trace_dist_untr),'avg Trace Distance from trained model: %f \n'%(trace_dist_train))
