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

#loading benchmark datasets (model can be trained with circuits of different lenghts if passed as list)
circuits_depth=5

benchmark_circ_path=os.getcwd()+'/src/rlnoise/bench_dataset'
model_path=os.getcwd()+'/src/rlnoise/saved_models/'
f = open(benchmark_circ_path+"/depth_%d.npz"%(circuits_depth),"rb")
tmp=np.load(f,allow_pickle=True)
train_set=tmp['train_set']
train_label=tmp['train_label']
val_set=tmp['val_set']
val_label=tmp['val_label']
f.close()

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
model = PPO(
    policy,
    circuit_env_training,
    policy_kwargs=policy_kwargs, 
    verbose=1,
    device='cuda'
)

#model=PPO.load(model_path+"rew_each_step_D7_box")

val_avg_rew_untrained,mea_untrained=(model_evaluation(val_set,val_label,circuit_env_training,model))

model.learn(1000, progress_bar=True) 
#model.save(model_path+"rew_each_step_D7_box_150k")

val_avg_rew_trained,mae_trained=(model_evaluation(val_set,val_label,circuit_env_training,model))
del model

print('avg reward from untrained model: %f\n'%(val_avg_rew_untrained),'avg reward from trained model: %f \n'%(val_avg_rew_trained))
print('avg MAE from untrained model: %f\n'%(mea_untrained*10),'avg MAE from trained model: %f \n'%(mae_trained*10))
