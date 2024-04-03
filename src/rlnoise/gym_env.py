import numpy as np
import copy
import json
import random
from pathlib import Path
from dataclasses import dataclass
from rlnoise.dataset import load_dataset
from rlnoise.circuit_representation import CircuitRepresentation
import gym
from gym import spaces
from qibo import gates

def gate_action_index(gate):
    if gate == 'epsilon_x':
        return 0
    if gate == 'epsilon_z':
        return 1
    if gate == gates.ResetChannel:
        return 2
    if gate == gates.DepolarizingChannel:
        return 3

# 
class DensityMatrixReward(object):
    """
    This class is used to define the reward function for the quantum circuit environment.
    It is possible to customize the reward function by passing a different metric function.
    It is also possible to use a different customized metric function.
    """
    def __init__(self, metric=lambda x,y: np.sqrt(np.abs(((x-y)**2)).mean())):
        self.metric = metric

    def __call__(self, circuit, target, final, alpha=1.):
        epsilon = 1e-10 # to avoid log(0)
        if final:
            circuit_dm = np.array(circuit().state())
            return -np.log(alpha * self.metric(circuit_dm, target) + epsilon)
        return 0.

@dataclass
class QuantumCircuit(gym.Env):
    '''
    Args: 
        dataset_file: path to the dataset file.
        config_file: path to the configuration file.
        reward: object, reward function, default is DensityMatrixReward().
    '''
    dataset_file: Path
    config_file: Path
    reward: object = DensityMatrixReward()
    kernel_size: int = None
    action_space_max_value: float = None
    only_depol: bool = None
    encoding_dim: int = 8
    circuits = None
    labels = None
    val_circuits = None
    val_labels = None
    rep = None
    circuit_number = None
    circuit_lenght = None
    padded_circuit = None

    def __post_init__(self):
        super().__init__()
        with open(self.config_file) as f:
            config = json.load(f)
        gym_env_params = config["gym_env"]
        self.kernel_size = gym_env_params['kernel_size']
        self.action_space_max_value = gym_env_params['action_space_max_value']
        self.only_depol = gym_env_params['enable_only_depolarizing']

        self.rep = CircuitRepresentation(self.config_file)

        self.circuits, self.labels, self.val_circuits, self.val_labels = load_dataset(self.dataset_file)

        if not self.kernel_size % 2 == 1:
            raise ValueError("Kernel_size must be an odd number.")
        
        self.position = None
        self.n_circ = len(self.circuits)
        self.n_qubits = self.circuits[0].shape[1]
        self.observation_space = spaces.Box(
            low = 0,
            high = 1,
            shape = (self.encoding_dim, self.n_qubits, self.kernel_size),
            dtype = np.float32
            )
        self.action_space = spaces.Box( 
            low=0, 
            high=1, 
            shape=(self.n_qubits, 4),    
            dtype=np.float32
            )
        
    def init_state(self, i=None):
        if i is None:
            i = random.randint(0, self.n_circ - 1)
        self.circuit_number = i
        self.circuit_lenght = self.circuits[i].shape[0]
        state = copy.deepcopy(self.circuits[i])
        state = state.transpose(2,1,0) 
        padding = np.zeros((self.encoding_dim, self.n_qubits, int(self.kernel_size/2)), dtype=np.float32)
        self.padded_circuit = np.concatenate((padding, state, padding), axis=2)
        return state, self.labels[i]
    
    def _get_obs(self):
        r = int(self.kernel_size/2)
        self.padded_circuit[:,:,r:-r] = self.current_state
        return np.asarray(self.padded_circuit[:,:,self.position:self.position+self.kernel_size], dtype=np.float32)
        
    def reset(self, i=None):
        self.position = 0
        self.current_state, self.current_target = self.init_state(i)
        return self._get_obs()
    
    def step(self, action):
        if self.only_depol:
            action[:, :3] = np.zeros((self.n_qubits, 3))
        self.current_state = self.rep.make_action(action, self.current_state, self.position)
        if self.position == self.circuit_lenght - 1:
            terminated = True
        else:
            self.position += 1
            terminated = False
        return self._get_obs(), self.reward(self.get_qibo_circuit(), self.current_target, terminated), terminated, None

    def get_qibo_circuit(self):
        return self.rep.array_to_circuit(self.current_state.transpose(2,1,0))