import numpy as np
from rlnoise.dataset import Dataset, CircuitRepresentation
from qibo.noise import DepolarizingError, NoiseModel
from qibo import gates
from rlnoise.rewards.rewards import FrequencyReward,DensityMatrixReward
from rlnoise.policy import CNNFeaturesExtractor
from rlnoise.envs.gym_env_v2 import QuantumCircuit
from stable_baselines3 import PPO

noise_model = NoiseModel()
lam = 0.2
noise_model.add(DepolarizingError(lam), gates.RZ)
noise_channel = gates.DepolarizingChannel((0,), lam=lam)
primitive_gates = ['RZ', 'RX']
channels = ['DepolarizingChannel']
reward = DensityMatrixReward()

rep = CircuitRepresentation(
    primitive_gates = primitive_gates,
    noise_channels = channels,
    shape = '2d'
)
depths=np.arange(5,20)

val_set=[]
val_label=[]
for i in depths:
    nqubits = 1
    depth = i
    ncirc = 100
    val_split = 0.2
    dataset = Dataset(
        n_circuits = ncirc,
        n_gates = depth,
        n_qubits = nqubits,
        representation = rep,
        clifford = True,
        shadows = True,
        noise_model = noise_model,
        mode = 'rep'
    )
    #dataset.save_bench_dataset(benchmark_circ_path)
    if i ==5:
        train_set=np.asarray(dataset.train_circuits)
        train_label=np.asarray(dataset.train_noisy_label)
    val_set.append(dataset.val_circuits)
    val_label.append(dataset.val_noisy_label)
    #print('train set shape: ', train_set.shape)



circuit_env_training = QuantumCircuit(
    circuits = train_set,
    noise_channel = noise_channel,
    representation = rep,
    labels = train_label,
    reward = reward,
    kernel_size=3
)
policy = "MlpPolicy"
policy_kwargs = dict(
    features_extractor_class = CNNFeaturesExtractor,
    features_extractor_kwargs = dict(
        features_dim = 64,
        filter_shape = (2, nqubits * rep.encoding_dim )
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



def model_evaluation(evaluation_circ,noise_channel,rep,evaluation_labels,reward,model):

    environment = QuantumCircuit(
    circuits = evaluation_circ,
    noise_channel = noise_channel,
    representation = rep,
    labels = evaluation_labels,
    reward = reward,
    kernel_size=3   
    )
    avg_rew=0
    n_circ=len(evaluation_circ)
    for i in range(n_circ):
        obs = environment.reset(i=i)
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = environment.step(action)
        untrained_circ = environment.get_qibo_circuit()
        dm_untrained=np.array(untrained_circ().state())
        avg_rew += rewards

    return avg_rew/n_circ

val_avg_rew_untrained=[]
val_avg_rew_trained=[]

for i in range(len(val_set)):
    val_avg_rew_untrained.append(model_evaluation(val_set[i],noise_channel,rep,val_label[i],reward,model))


model.learn(5000, progress_bar=True) 


for i in range(len(val_set)):
    val_avg_rew_trained.append(model_evaluation(val_set[i],noise_channel,rep,val_label[i],reward,model))

print(val_avg_rew_untrained,'\n',val_avg_rew_trained)

print('The RL model was trained on %d circuits with depth %d'%(train_set.shape[0],train_set.shape[1]))
for i in range(len(val_set)):
    print('The validation performed on %d circuits with depth %d has produced this rewards: '%(val_set[i].shape[0],val_set[i].shape[1]))
    print('avg reward from untrained model: %f\n'%(val_avg_rew_untrained[i]),'avg reward from trained model: %f \n'%(val_avg_rew_trained[i]))
#non so come salvare lista di array con dimensioni diverse tra loro. serve per fare benchmark sempre sugli stessi circuiti
#per allenare il modello su circuiti di lunghezza variabili va modificata la logica del gym environment
#la funzione model evaluation forse puo essere aggiunta in utility
#tutto questo funziona con gym_env_v2 ma non gym_env 