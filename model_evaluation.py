import numpy as np
import os
from rlnoise.dataset import Dataset, CircuitRepresentation
from qibo.noise import DepolarizingError, NoiseModel
from qibo import gates
from rlnoise.rewards.rewards import FrequencyReward,DensityMatrixReward
from rlnoise.policy import CNNFeaturesExtractor
from rlnoise.gym_env import QuantumCircuit
from stable_baselines3 import PPO
from rlnoise.utils import model_evaluation

#loading benchmark datasets
benchmark_circ_path=os.getcwd()+'/src/rlnoise/bench_dataset'
f = open(benchmark_circ_path+"/depth_15.npz","rb")
tmp=np.load(f,allow_pickle=True)
train_set=tmp['train_set']
train_label=tmp['train_label']
val_set=tmp['val_set']
val_label=tmp['val_label']
f.close()

#Setting up training env and policy model
nqubits=2
noise_model = NoiseModel()
lam = 0.2
noise_model.add(DepolarizingError(lam), gates.RZ)
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
print('Train dataset circuit shape: ',train_set.shape)
print('train label shape: ',train_label.shape)

val_avg_rew_untrained=(model_evaluation(val_set,val_label,circuit_env_training,model))

model.learn(10000, progress_bar=True) 

val_avg_rew_trained=(model_evaluation(val_set,val_label,circuit_env_training,model))

print('The RL model was trained on %d circuits with depth %d'%(train_set.shape[0],15))
print('The validation performed on %d circuits with depth %d has produced this rewards: '%(val_set.shape[0],15))
print('avg reward from untrained model: %f\n'%(val_avg_rew_untrained),'avg reward from trained model: %f \n'%(val_avg_rew_trained))

#per allenare il modello su circuiti di lunghezza variabili va modificata la logica del gym environment

#far funzionre il training con gym env: fatto
#dividere il test in 2 file, model.py e evaluation.py: fatto
#pulire test gym env solo per testare che l'ambiente funzioni, creo/carico dataset con 2 circ: fatto
