from rlnoise.neural_network import CNNFeaturesExtractor
from rlnoise.callback import CustomCallback
from stable_baselines3 import PPO
from rlnoise.gym_env import QuantumCircuit
import numpy as np
from qibo import Circuit
import json

class Agent(object):

    def __init__(self, config_file, env: QuantumCircuit):

        self.callback = CustomCallback(config_path = config_file, env = env)  

        with open(config_file) as f:
            config = json.load(f)

        nqubits = config["dataset"]["qubits"]
        policy = config["agent"]["policy"]
        features_dim = config["agent"]["features_dim"]
        n_steps = config["agent"]["nn_update_steps"]
        batch_size = config["agent"]["batch_size"]
        self.env = env

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

    def train(self, n_steps):
        self.model.learn(total_timesteps=n_steps, progress_bar=True, callback=self.callback) 

    def apply(self, circuit, return_qibo_circuit = True):
        if isinstance(circuit, Circuit):
            circuit = self.env.rep.circuit_to_array(circuit)

        circuit = np.expand_dims(circuit, axis=0)
        circuit_env = QuantumCircuit(config_file = self.env.config_file, dataset_file = None, circuits = circuit)

        terminated = False
        obs, _ = circuit_env.reset(i=0)
        while not terminated:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = circuit_env.step(action, reward=False)
        if not return_qibo_circuit:
            return circuit_env.current_state.transpose(2,1,0)
        return circuit_env.get_qibo_circuit()
