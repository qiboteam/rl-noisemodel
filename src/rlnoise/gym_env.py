import numpy as np
import copy
import json
import random
from pathlib import Path
from dataclasses import dataclass
from rlnoise.dataset import load_dataset
from rlnoise.circuit_representation import CircuitRepresentation
from rlnoise.utils import mse, trace_distance, compute_fidelity
import gymnasium
from gymnasium import spaces
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

class DensityMatrixReward(object):
    """
    This class is used to define the reward function for the quantum circuit environment.
    It is possible to customize the reward function by passing a different metric function.
    It is also possible to use a different customized metric function.
    """
    def __init__(self, metric="mse", function="log", alpha=1.):
        if metric == "mse":
            self.metric = mse
        elif metric == "fidelity":
            self.metric = compute_fidelity
        elif metric == "trace_distance":
            self.metric = trace_distance
        else:
            raise ValueError("Invalid metric function.")

        if function == "log":
            self.function = lambda x: -np.log(alpha * x + 1e-15)
        elif function == "linear":
            self.function = lambda x: alpha * x
        else:
            raise ValueError("Invalid function.")
        

    def __call__(self, circuit, target, final):
        if final:
            circuit_dm = np.array(circuit().state())
            return self.function(self.metric(circuit_dm, target))
        return 0.

@dataclass
class QuantumCircuit(gymnasium.Env):
    '''
    Args: 
        config_file: path to the configuration file.
    '''
    config_file: Path 
    dataset_file: Path = None
    circuits: np.ndarray = None
    labels = None
    val_split = None
    kernel_size: int = None
    action_space_max_value: float = None
    only_depol: bool = None
    encoding_dim: int = 8
    rep = None
    circuit_number = None
    circuit_lenght = None
    padded_circuit = None

    def __post_init__(self):
        super().__init__()
        with open(self.config_file) as f:
            config = json.load(f)
        gym_env_params = config["gym_env"]
        reward_params = gym_env_params["reward"]
        self.kernel_size = gym_env_params['kernel_size']
        self.only_depol = gym_env_params['enable_only_depolarizing']
        if self.val_split is None:
            self.val_split = gym_env_params['val_split']

        self.rep = CircuitRepresentation(self.config_file)

        # Define the reward function
        reward_matric = reward_params["metric"]
        reward_func = reward_params["function"]
        reward_alpha = reward_params["alpha"]
        self.reward = DensityMatrixReward(metric=reward_matric, function=reward_func, alpha=reward_alpha)

        if self.circuits is None:
            self.circuits, self.labels = load_dataset(self.dataset_file)
        
        if not self.kernel_size % 2 == 1:
            raise ValueError("Kernel_size must be an odd number.")
        
        self.position = None
        self.n_circ = len(self.circuits)
        self.n_circ_train = int((1 - self.val_split) * self.n_circ)
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
                i = random.randint(0, self.n_circ_train)
        self.circuit_number = i
        self.circuit_lenght = self.circuits[i].shape[0]
        state = copy.deepcopy(self.circuits[i])
        state = state.transpose(2,1,0) 
        padding = np.zeros((self.encoding_dim, self.n_qubits, int(self.kernel_size/2)), dtype=np.float32)
        self.padded_circuit = np.concatenate((padding, state, padding), axis=2)
        if self.labels is None:
            return state, None
        return state, self.labels[i]
    
    def _get_obs(self):
        r = int(self.kernel_size/2)
        self.padded_circuit[:,:,r:-r] = self.current_state
        return np.asarray(self.padded_circuit[:,:,self.position:self.position+self.kernel_size], dtype=np.float32)
        
    def reset(self, i=None, seed=None):
        self.position = 0
        self.current_state, self.current_target = self.init_state(i)
        return self._get_obs(), None
    
    def step(self, action, reward = True):
        if self.only_depol:
            action[:, :3] = np.zeros((self.n_qubits, 3))
        self.current_state = self.rep.make_action(action, self.current_state, self.position)
        if self.position >= self.circuit_lenght - 1:
            terminated = True
        else:
            self.position += 1
            terminated = False
        if reward:
            reward = self.reward(self.get_qibo_circuit(), self.current_target, terminated)
        else:
            reward = 0.
        # Observation, Reward, Terminated, Truncated (always False), Info (Dict)
        return self._get_obs(), reward, terminated, False, {}

    def get_qibo_circuit(self):
        return self.rep.rep_to_circuit(self.current_state.transpose(2,1,0))