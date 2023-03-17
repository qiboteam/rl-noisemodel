import numpy as np
import gym, random
from gym import spaces
from gym.spaces import Dict, Box, Discrete, MultiBinary
from qibo import gates
from qibo.models import Circuit
from copy import deepcopy
from utils import truncated_moments_matching



# currently working just for single qubit circuits
# TO DO:
# - Adapt to multi-qubits circuits
# - Implement batches
# - Write a reward that makes sense
# - Adapt it for working with the 3d representation
class QuantumCircuit(gym.Env):
    
    def __init__(self, circuits, noise_channel, representation, labels, reward_f):
        super(QuantumCircuit, self).__init__()

        self.circuits = circuits[:,np.newaxis,:,:]
        self.n_circ = self.circuits.shape[0]
        self.n_moments = self.circuits.shape[2]
        self.noise_channel = noise_channel
        self.rep = representation
        self.n_gate_types = len(self.rep.gate2index)
        self.n_channel_types = len(self.rep.channel2index)
        self.labels = labels

        shape = list(self.circuits.shape[1:])
        assert shape[-1] % (self.rep.encoding_dim) == 0
        self.n_qubits = int(shape[-1] / self.rep.encoding_dim)
        shape[-1] += 1

        assert self.n_channel_types == 1, "Multiple possible channels not implemented yet"
        
        self.observation_space = spaces.Box(
            low = 0,
            high = 1,
            shape = tuple(shape),
            dtype = np.float32
        )

        #self.action_space = spaces.Discrete(2)
        self.action_space = spaces.MultiBinary(self.n_qubits * self.n_channel_types)
        
        self.current_state = self.init_state()

    def init_state(self, i=None):
        # initialize the state
        if i is None:
            i = random.randint(0, self.n_circ - 1)
        state = np.hstack(
            ( self.circuits[i].squeeze(0), np.zeros((self.n_moments, 1)) )
        )
        state[0,-1] = 1
        state = state[np.newaxis,:,:]
        return state
    
    def _get_obs(self):
        return self.current_state

    def _get_info(self):
        return { 'state': self._get_obs() } 

    def reset(self, i=None):
        self.current_state = self.init_state(i)
        return self._get_obs()

    def step(self, action):
        #print('> State:\n', self._get_obs())
        position = self.get_position()
        action = action.reshape(self.n_qubits, -1)
        #print(f'> Position: {position}, Action: {action}')
        for q, a in enumerate(action):
            if a == 1:
                idx = q * self.rep.encoding_dim + self.n_gate_types + 1
                self.current_state[0, position, idx] = a
                self.current_state[0, position, idx + 1] = self.noise_channel.init_kwargs['lam']
        #print(f'> New State: \n{self._get_obs()}')
        if position == self.n_moments - 1:
            # compute final reward
            reward = self.compute_reward()
            terminated = True
        else:
            # update position
            self.current_state[0, position, -1] = 0
            self.current_state[0, position + 1, -1] = 1
            reward = 0
            terminated = False
        return self._get_obs(), reward, terminated, self._get_info()

    def render(self):
        print(self.get_qibo_circuit().draw(), end='\r')
            
    def get_position(self):
        return (self.current_state[:,:,-1] == 1).nonzero()[-1]

    def get_qibo_circuit(self):
        return self.rep.array_to_circuit(self.current_state[0][:,:-1])

    def compute_reward(self):
        # TO BE IMPLEMENTED
        c = self.get_qibo_circuit()
        #print(c.draw())
        c.add([gates.M(i) for i in range(self.n_qubits)])
        freq = c(nshots=10000).frequencies()
        #print(freq)
        zero = (freq['0'] - self.labels['0'])/self.labels['0']
        one = (freq['1'] - self.labels['1'])/self.labels['1']
        r = - np.sqrt(zero**2 + one**2)
        return r
        #return self.reward_f(c)
        #return random.randint(0,1)
