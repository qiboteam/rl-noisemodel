from rlnoise.neural_network import CNNFeaturesExtractor
from rlnoise.callback import CustomCallback
from stable_baselines3 import PPO
from rlnoise.gym_env import QuantumCircuit
from rlnoise.utils import compute_fidelity, mse, trace_distance
import numpy as np
from qibo import Circuit
import json

class Agent(object):
    '''Agent class that uses the PPO algorithm from stable_baselines3 to train a policy.
    The agent can be used to apply the policy to a circuit and return the resulting circuit.'''
    def __init__(self, config_file, env: QuantumCircuit, model_file_path = None):

        self.callback = CustomCallback(config_path = config_file, env = env)  

        with open(config_file) as f:
            config = json.load(f)

        nqubits = config["dataset"]["qubits"]
        policy = config["agent"]["policy"]
        filter_size = config["agent"]["filter_size"]
        conv_filters = config["agent"]["n_filters"]
        features_dim = config["agent"]["features_dim"]
        n_steps = config["agent"]["nn_update_steps"]
        batch_size = config["agent"]["batch_size"]
        self.env = env

        policy_kwargs = dict(
        features_extractor_class = CNNFeaturesExtractor,
        features_extractor_kwargs = dict(
            features_dim = features_dim,
            filter_shape = (nqubits, filter_size),
            n_filters = conv_filters
        ),
        net_arch=dict(pi=[32, 32], vf=[32, 32])
        )
        if model_file_path is not None:
            self.load(model_file_path)
        else:
            self.model = PPO(
            policy,
            env,
            policy_kwargs = policy_kwargs,
            n_steps = n_steps,
            batch_size = batch_size
            )

    def train(self, n_steps):
        '''Train the agent for n_steps using the PPO algorithm.'''
        self.model.learn(total_timesteps=n_steps, progress_bar=True, callback=self.callback) 

    def load(self, file_path):
        '''Load a model from a file.'''
        self.model = PPO.load(file_path)

    def apply(self, circuit, return_qibo_circuit = True):
        '''Apply the policy to a circuit and return the resulting circuit.'''
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
    
    def apply_rb_dataset(self, rb_dataset, verbose = False):
        '''Apply the policy to a dataset of circuits used for RB and return the fidelity and trace distance.'''
        dataset = np.load(rb_dataset, allow_pickle=True)
        circuits = dataset["circuits"]
        labels = dataset["labels"]
        rep = self.env.rep
        final_result = []

        print("preprocessing circuits...")
        final_circuits = {}
        for same_len_circuits in circuits:
            for rep_c in same_len_circuits:
                c = rep.rep_to_circuit(rep_c)
                depth = c.depth
                if depth not in final_circuits.keys():
                    final_circuits[depth] = []
                final_circuits[depth].append(c)
        circuits = final_circuits

        print("Evaluating RL agent...")
        for label_index, circs in enumerate(circuits.values()):
            depth = circs[0].depth
            if verbose:
                print(f'> Looping over circuits of depth: {depth}')
            fidelity = []
            mse_ = []
            for i, c in enumerate(circs):
                noisy_circuit = self.apply(c)    
                dm_noise = noisy_circuit().state()
                fidelity.append(compute_fidelity(labels[label_index][i], dm_noise))
                mse_.append(mse(labels[label_index][i], dm_noise))
            fidelity = np.array(fidelity)
            mse_ = np.array(mse_)
            result = np.array([(
                depth,
                fidelity.mean(),
                fidelity.std(),
                mse_.mean(),
                mse_.std(),
                )],
                dtype=[('depth','<f4'),
                        ('fidelity','<f4'),
                        ('fidelity_std','<f4'),
                        ('mse','<f4'),
                        ('mse_std','<f4'),
                    ])
            final_result.append(result)
        
        return np.asarray(final_result)
    
    def apply_eval_dataset(self, dataset, verbose = True):
        '''Apply the policy to a dataset of circuits used for testing and return the fidelity, trace distance and the dms.'''
        dataset = np.load(dataset, allow_pickle=True)
        circuits = dataset["circuits"]
        labels = dataset["labels"]
        final_result = []
        dms = []
        avg_fidelity = 0.
        avg_mse = 0.
        avg_trace = 0. 
        for i, c in enumerate(circuits):
            noisy_circuit = self.apply(c)    
            dm_noise = noisy_circuit().state()
            dms.append(dm_noise)
            fidelity = compute_fidelity(labels[i], dm_noise)
            trace_d = trace_distance(labels[i], dm_noise)
            mse_ = mse(labels[i], dm_noise)
            result = np.array([(
                fidelity,
                mse_,
                trace_d,
            )],
                dtype=[ ('fidelity','<f4'),
                        ('mse','<f4'),
                        ('trace_distance','<f4'),
                    ])
            final_result.append(result)
            avg_fidelity += fidelity
            avg_mse += mse_
        if verbose:
            print("Avg fidelity: ", avg_fidelity/len(circuits))
            print("Avg mse: ", avg_mse/len(circuits))
        return np.asarray(final_result), dms
            
            
