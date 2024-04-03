from rlnoise.policy import CNNFeaturesExtractor, CustomCallback
from stable_baselines3 import PPO
import json

class Agent(object):

    def __init__(self, config_file, env):

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

        self.callback = CustomCallback(check_freq=2500,
                        dataset=tmp,
                        train_environment=circuit_env_training,
                        verbose=True,
                        result_filename=results_filename,
                        )     
