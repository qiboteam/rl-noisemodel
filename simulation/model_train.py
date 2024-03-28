from rlnoise.policy import CNNFeaturesExtractor,CustomCallback
from rlnoise.gym_env import QuantumCircuit
from stable_baselines3 import PPO

dataset_file = "simulation/dataset/3q_high_noise.npz"
config_file = "simulation/config/config_3q_high_noise.json"

circuit_env = QuantumCircuit(
    dataset_file=dataset_file,
    config_file=config_file,
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
n_steps=256,
)
#                             #STANDARD TRAINING

callback=CustomCallback(check_freq=2500,
                        dataset=tmp,
                        train_environment=circuit_env_training,
                        verbose=True,
                        result_filename=results_filename,
                        )                                          

model.learn(total_timesteps=200000, progress_bar=True, callback=callback)


