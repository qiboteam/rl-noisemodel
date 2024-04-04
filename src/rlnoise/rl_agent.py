from rlnoise.neural_network import CNNFeaturesExtractor
from rlnoise.callback import CustomCallback
from stable_baselines3 import PPO
import json

class Agent(object):

    def __init__(self, config_file, env):

        self.callback = CustomCallback(config_file = config_file)  

        with open(config_file) as f:
            config = json.load(f)

        nqubits = config["dataset"]["qubits"]
        policy = config["agent"]["policy"]
        features_dim = config["agent"]["features_dim"]
        n_steps = config["agent"]["update_steps"]
        batch_size = config["agent"]["batch_size"]

        policy_kwargs = dict(
        features_extractor_class = CNNFeaturesExtractor,
        features_extractor_kwargs = dict(
            features_dim = features_dim,
            filter_shape = (nqubits, 1)
        ),
        net_arch=dict(pi=[32, 32], vf=[32, 32])
        )

        self.model = PPO(
        policy,
        env,
        policy_kwargs = policy_kwargs,
        n_steps = n_steps,
        batch_size = batch_size
        )

    def train(self, n_steps, callback=True):
        self.model.learn(total_timesteps=n_steps, progress_bar=True, callback=self.callback) 
