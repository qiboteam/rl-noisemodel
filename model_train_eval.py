import numpy as np
import os
from rlnoise.dataset import Dataset, CircuitRepresentation
from qibo.noise import DepolarizingError, NoiseModel
from qibo import gates
from rlnoise.rewards.rewards import FrequencyReward,DensityMatrixReward
from rlnoise.policy import CNNFeaturesExtractor
from rlnoise.gym_env import QuantumCircuit
from stable_baselines3 import PPO,DQN
from rlnoise.utils import model_evaluation

#loading benchmark datasets
circuits_depth=7
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
nqubits=2
noise_model = NoiseModel()
lam = 0.01
lamCZ=0.1
noise_model.add(DepolarizingError(lam), gates.RZ)
noise_model.add(DepolarizingError(lamCZ), gates.CZ)
noise_channel = gates.DepolarizingChannel((0,), lam=lam)
primitive_gates = ['RZ', 'RX','CZ']
channels = ['DepolarizingChannel']
reward = DensityMatrixReward()
kernel_size=3

rep = CircuitRepresentation(
    primitive_gates = primitive_gates,
    noise_channels = channels,
    shape = '3d'
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
)
print('Train dataset circuit shape: ',train_set.shape,'number of qubits: ',train_set[0].shape)
print('train label shape: ',train_label.shape)
#model=PPO.load(model_path+"trained_model_D7_box_600k")
val_avg_rew_untrained=(model_evaluation(val_set,val_label,circuit_env_training,model))

model.learn(50000, progress_bar=True) 
model.save(model_path+"rew_each_step_D7_box")
val_avg_rew_trained=(model_evaluation(val_set,val_label,circuit_env_training,model))
del model
print('The RL model was trained on %d circuits with depth %d'%(train_set.shape[0],30))
print('The validation performed on %d circuits with depth %d has produced this rewards: '%(val_set.shape[0],30))
print('avg reward from untrained model: %f\n'%(val_avg_rew_untrained),'avg reward from trained model: %f \n'%(val_avg_rew_trained))

