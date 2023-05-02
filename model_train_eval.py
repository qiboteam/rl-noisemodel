import numpy as np
import os
from rlnoise.datasetv2 import Dataset, CircuitRepresentation
from qibo import gates
from rlnoise.rewards.rewards import FrequencyReward,DensityMatrixReward
from rlnoise.policy import CNNFeaturesExtractor
from rlnoise.gym_env import QuantumCircuit
from stable_baselines3 import PPO,DQN,DDPG #not bad
from stable_baselines3 import DQN,A2C,TD3
from rlnoise.utils import model_evaluation
from rlnoise.CustomNoise import CustomNoiseModel
from rlnoise.MlpPolicy import MlPFeaturesExtractor
#loading benchmark datasets (model can be trained with circuits of different lenghts if passed as list)
circuits_depth=5

benchmark_circ_path=os.getcwd()+'/src/rlnoise/bench_dataset'
model_path=os.getcwd()+'/src/rlnoise/saved_models/'

bench_results_path=os.getcwd()+'/src/rlnoise/bench_results'
f = open(benchmark_circ_path+"/depth_%dDep_only_1Q.npz"%(circuits_depth),"rb")
tmp=np.load(f,allow_pickle=True)
train_set=tmp['train_set']
train_label=tmp['train_label']
val_set=tmp['val_set']
val_label=tmp['val_label']
f.close()

#Setting up training env and policy model
nqubits=1
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
    kernel_size=kernel_size
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
'''                                                 #TRAIN & TEST ON SAME DEPTH
time_step=[10000,15000,20000,25000,30000,40000]
print(str(time_step))
for total_timesteps in time_step:
    model = PPO(
    policy,
    circuit_env_training,
    policy_kwargs=policy_kwargs, 
    verbose=0,
    )
    val_avg_rew_untrained,mae_untrained,trace_dist_untr=(model_evaluation(val_set,val_label,circuit_env_training,model))
    Rew_Mae_TraceD_untrained.append([val_avg_rew_untrained,mae_untrained,trace_dist_untr])

    model.learn(total_timesteps, progress_bar=True) 

    val_avg_rew_trained,mae_trained,trace_dist_train=(model_evaluation(val_set,val_label,circuit_env_training,model))
    Rew_Mae_TraceD_trained.append([val_avg_rew_trained,mae_trained,trace_dist_train])
    del model
Rew_Mae_TraceD_untrained=np.array(Rew_Mae_TraceD_untrained)
Rew_Mae_TraceD_trained=np.array(Rew_Mae_TraceD_trained)

f = open(bench_results_path+"/Depol_only_D5_Q1_K3_SR-off_ts"+str(time_step),"wb")
np.savez(f,untrained=Rew_Mae_TraceD_untrained,trained=Rew_Mae_TraceD_trained)
f.close()
'''
model=PPO.load(model_path+"D5_SR_K3_1Q_Dep_only_30k")
depth_list=[5,10,15,20]
for d in depth_list:
    f = open(benchmark_circ_path+"/depth_%dDep_only_1Q.npz"%(d),"rb")
    tmp=np.load(f,allow_pickle=True)
    val_set=tmp['val_set']
    val_label=tmp['val_label']
    f.close()
    val_avg_rew_trained,mae_trained,trace_dist_train=(model_evaluation(val_set,val_label,circuit_env_training,model))
    Rew_Mae_TraceD_trained.append([val_avg_rew_trained,mae_trained,trace_dist_train])

Rew_Mae_TraceD_trained=np.array(Rew_Mae_TraceD_trained)
f = open(bench_results_path+"/Depol_only_D5(train)_Q1_K3_SR-off_Test"+str(depth_list),"wb")
np.savez(f,trained=Rew_Mae_TraceD_trained)
f.close()
#print('avg reward from untrained model: %f\n'%(val_avg_rew_untrained),'avg reward from trained model: %f \n'%(val_avg_rew_trained))
#print('avg MAE from untrained model: %f\n'%(mea_untrained*10),'avg MAE from trained model: %f \n'%(mae_trained*10))
#print('avg Trace Distance from untrained model: %f\n'%(trace_dist_untr),'avg Trace Distance from trained model: %f \n'%(trace_dist_train))




#DEBUG CONSIDERATIONS:
#1) rep_to _circuit seems to work well with 1 qubit and only depolarizing noise. It seems to work also for more qubit (max 3) and with thermal and also coherent noise.
#2) CustumNoise is working properly on 1 qubit with depolarizing and/or tehrmal. Also coherent noise is added correctly and NEEDS TO BE ADDDED A METHOD TO GET THE NOISY_REPRESENTATION also to better test 1)
