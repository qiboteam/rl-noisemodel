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
from rlnoise.utils import RL_NoiseModel
from rlnoise.utils import model_evaluation, RB_evaluation
import torch
from rlnoise.metrics import compute_fidelity

current_path = Path(__file__).parent
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=f"{current_path}/src/rlnoise/config.json")
parser.add_argument('--dataset', type=str, default=f"{current_path}/src/rlnoise/model_folder/test_set_enhanced_3Q_len500.npz")
parser.add_argument('--output', type=str, default=f"{current_path}/src/rlnoise/model_folder")
args = parser.parse_args()

#IMPLEMENTING A CUSTUM POLICY NETWORK (e.g. increasing dimension of value network) COULD BE AN IDEA
results_filename = f'{args.output}/train_results_mse_tanh'

#loading benchmark datasets (model can be trained with circuits of different lenghts if passed as list)
tmp = np.load(args.dataset, allow_pickle=True)
train_set = copy.deepcopy(tmp['train_set'])
train_label = copy.deepcopy(tmp['train_label'])
val_set = copy.deepcopy(tmp['val_set'])
val_label = copy.deepcopy(tmp['val_label'])

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

# print((model.policy.action_net))
model.learn(total_timesteps=500000, progress_bar=True, callback=callback)

# agent = RL_NoiseModel(model, rep)

# for i in [0,20,7,9]:
#     print(val_set.shape)
#     test_circ = rep.rep_to_circuit(val_set[i])
#     test_label = val_label[i]
#     rl_noisy_circ = agent.apply(test_circ)
#     print(compute_fidelity(rl_noisy_circ().state(), test_label))

# TESTING A PREVIOUSLY TRAINED MODEL ON DIFFERENT DEPTHS AND COMPARING WITH RB AND WITH UNTRAINED MODEL

# results_list_untrained=[]
# results_list_trained = []
# result_RB_list = []
# model_trained = PPO.load("src/rlnoise/simulation_phase/1Q_training_new/1Q_D7_Simulation546000.zip")

# nqubits=1
# n_circuit_in_dataset=50
# depth_list=np.arange(3,31,3)
# benchmark_circ_path = 'src/rlnoise/simulation_phase/RB/1Q/dataset/'
# bench_results_path = 'src/rlnoise/simulation_phase/RB/1Q/results'
# result_filename='comparison2_results_1Q.npz'
# for d in depth_list:
#     dataset_name='RB_set'+'_D%d_%dQ_len%d.npz'%(d,nqubits,n_circuit_in_dataset)
#     with open(benchmark_circ_path+dataset_name,"rb") as f:
#         tmp=np.load(f,allow_pickle=True)
#         val_set=tmp['clean_rep']
#         val_label=tmp['label']
#     results_trained_model = model_evaluation(val_set,val_label,model_trained,reward=reward,representation=rep)
#     results_RB = RB_evaluation(lambda_RB=0.08,circ_representation=val_set,target_label=val_label)
#     results_list_trained.append(results_trained_model)
#     result_RB_list.append(results_RB)
# model_results = np.array(results_list_trained)
# rb_results = np.array(result_RB_list)

# with open(bench_results_path+result_filename,"wb") as f:
#     np.savez(f,
#              trained=model_results,
#              RB=rb_results)


        #    TRAINING ON DIFFERENT DATASET SIZE (Evaluating the best dataset size for overfitting)

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


