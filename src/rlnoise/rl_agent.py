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

    def apply(self, circuit):
        if isinstance(circuit, Circuit):
            circuit = self.rep.circuit_to_array(circuit)
        circ_len = circuit.shape[0]
        padding = np.zeros((8, self.env.nqubits, int(self.kernel_size/2)), dtype=np.float32)
        self.padded_circuit = np.concatenate((padding, state, padding), axis=2)

        for pos in range(circuit.shape[-1]):
            l_flag = pos - self.ker_radius < 0
            r_flag = pos + self.ker_radius > circuit.shape[2] - 1
            if l_flag and not r_flag:
                pad_cel = np.zeros((circuit.shape[0], circuit.shape[1], np.abs(pos - self.ker_radius)))
                observation = np.concatenate((pad_cel, circuit[:, :, :pos + self.ker_radius + 1]), axis=2)
            elif not l_flag and r_flag:
                pad_cel = np.zeros((circuit.shape[0], circuit.shape[1], np.abs(pos + self.ker_radius - circuit.shape[-1] + 1)))
                observation = np.concatenate((circuit[:, :, pos - self.ker_radius:], pad_cel), axis=2)
            elif l_flag and r_flag:
                l_pad = np.zeros((circuit.shape[0], circuit.shape[1], np.abs(pos - self.ker_radius)))
                l_obs = np.concatenate((l_pad, circuit[:, :, :pos + self.ker_radius + 1]), axis=2)
                r_pad = np.zeros((circuit.shape[0], circuit.shape[1], np.abs(pos + self.ker_radius - circuit.shape[-1] + 1)))
                r_obs = np.concatenate((circuit[:, :, pos - self.ker_radius:], r_pad), axis=2)
                observation = np.concatenate((l_obs, r_obs), axis=2)
            else:
                observation = circuit[:, :, pos - self.ker_radius: pos + self.ker_radius + 1]   
            action, _ = self.agent.predict(observation, deterministic=True)
            circuit = self.rep.make_action(action=action, circuit=circuit, position=pos)
        return self.rep.rep_to_circuit(np.transpose(circuit, axes=[2,1,0]))
