import numpy as np
import gym, random
from gym import spaces
from qibo import gates
from qibo.models import Circuit
from copy import deepcopy
from rlnoise.rewards.density_matrix_reward import dm_reward_stablebaselines, step_reward_stablebaselines

# currently working just for single qubit circuits
# TO DO:
# - Adapt to multi-qubits circuits
# - Implement batches
# - Write a reward that makes sense
# - Adapt it for working with the 3d representation


class QuantumCircuit(gym.Env):
    
    def __init__(self, circuits, noise_channel, representation, labels, reward):
        super(QuantumCircuit, self).__init__()

        self.circuits = circuits[:,np.newaxis,:,:]
        self.n_circ = self.circuits.shape[0]
        self.n_moments = self.circuits.shape[2]
        self.noise_channel = noise_channel
        self.rep = representation
        self.n_gate_types = len(self.rep.gate2index)
        self.n_channel_types = len(self.rep.channel2index)
        self.labels = labels
        self.reward = reward

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
        self.current_state, self.current_target = self.init_state()

    def init_state(self, i=None):
        # initialize the state
        if i is None:
            i = random.randint(0, self.n_circ - 1)
        state = np.hstack(
            ( self.circuits[i].squeeze(0), np.zeros((self.n_moments, 1)) )
        )
        state[0,-1] = 1
        state = state[np.newaxis,:,:]
        return state, self.labels[i]
    
    def _get_obs(self):
        return self.current_state #current_state.shape= (1, depth, 6 if only 1qubit )

    def _get_info(self):
        return { 'state': self._get_obs() } 

    def reset(self, i=None):
        self.current_state, self.current_target = self.init_state(i)
        return self._get_obs()

    def step(self, action):
        #print('> State:\n', self._get_obs())
        position = self.get_position()
        action = action.reshape(self.n_qubits, -1) #action.shape=(num_qubits, 1)
        
        #print(f'> Position: {position}, Action: {action} , Action shape: {action.shape}')
        for q, a in enumerate(action):
            if a == 1:
                copy = self.current_state.copy()
                # might be better to use self.rep.array_to_gate()
                #idx = q * self.rep.encoding_dim + self.n_gate_types + 1 #idx selects the column of the noise (for 1 qubit and 2 type_of_gates is the 3rd)
                #self.current_state[0, position, idx] = a #current_state.shape= (1, depth, 6 if only 1qubit )
                #self.current_state[0, position, idx + 1] = self.noise_channel.init_kwargs['lam']
                idx = q * self.rep.encoding_dim
                self.current_state[0, position, idx:idx+self.rep.encoding_dim] += self.rep.gate_to_array(self.noise_channel)
        #print(f'> New State: \n{self._get_obs()}')
        if position == self.n_moments - 1:
            # compute final reward
            terminated = True
        else:
            # update position
            self.current_state[0, position, -1] = 0
            self.current_state[0, position + 1, -1] = 1
            terminated = False
        reward = self.reward(self.get_qibo_circuit(), self.current_target, terminated)
        return self._get_obs(), reward, terminated, self._get_info()

    def render(self):
        print(self.get_qibo_circuit().draw(), end='\r')
            
    def get_position(self):
        return (self.current_state[:,:,-1] == 1).nonzero()[-1]

    def get_qibo_circuit(self):
        return self.rep.array_to_circuit(self.current_state[0][:,:-1])


    
