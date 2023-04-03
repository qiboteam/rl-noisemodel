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
    
    def __init__(self, circuits, noise_channel, representation, labels, reward, kernel_size=3):
        super(QuantumCircuit, self).__init__()
        assert kernel_size%2==1, "Kernel_size must be odd"
        self.circuits = circuits[:,np.newaxis,:,:]
        self.n_circ = self.circuits.shape[0]
        self.n_moments = self.circuits.shape[2]
        self.noise_channel = noise_channel
        self.rep = representation
        self.n_gate_types = len(self.rep.gate2index)
        self.n_channel_types = len(self.rep.channel2index)
        self.labels = labels
        self.reward = reward
        self.padding_len=int(kernel_size/2)
        shape = list(self.circuits.shape[1:])
        assert shape[-1] % (self.rep.encoding_dim) == 0
        self.n_qubits = int(shape[-1] / self.rep.encoding_dim)
        self.circuit_shape=shape

        assert self.n_channel_types == 1, "Multiple possible channels not implemented yet"
        
        self.observation_space = spaces.Box(
            low = 0,
            high = 1,
            shape = (shape[0],2*self.padding_len,shape[-1]),
            dtype = np.float32
        )
        self.action_space = spaces.MultiBinary(self.n_qubits * self.n_channel_types)
        self.current_state = None
    
    def _get_obs(self):
        return self.circuit_padding[:,self.pos-self.padding_len:self.pos+self.padding_len,:]

    def _get_info(self):
        return { 'state': self._get_obs() } 

    def reset(self, i=None):
        if i is None:
            i = random.randint(0, self.n_circ - 1)
        self.sample = i
        self.pos = self.padding_len
        state = self.circuits[i]
        self.current_target=self.labels[i]
        padding = np.zeros((1, self.padding_len, self.circuit_shape[-1]), dtype=np.float32)
        self.circuit_padding = np.concatenate((padding,state,padding), axis=1)
        
        return self._get_obs()

    def step(self, action):
        #print('> State:\n', self._get_obs())
        action = action.reshape(self.n_qubits, -1) #action.shape=(num_qubits, 1)        
        #print(f'> Position: {position}, Action: {action} , Action shape: {action.shape}')
        for q, a in enumerate(action):
            if a == 1:
                idx = q * self.rep.encoding_dim
                self.circuit_padding[0, self.pos, idx:idx+self.rep.encoding_dim] += self.rep.gate_to_array(self.noise_channel) 
        #print(f'> New State: \n{self._get_obs()}')
        terminated = False
        if self.pos == self.padding_len+self.n_moments - 1:
            # compute final reward
            terminated = True
        self.pos+=1
        return self._get_obs(), self.reward(self.get_qibo_circuit(), self.current_target, terminated), terminated, self._get_info()

    def render(self):
        print(self.get_qibo_circuit().draw(), end='\r')
            
    def get_position(self):
        return (self.current_state[:,:,-1] == 1).nonzero()[-1]

    def get_qibo_circuit(self):
        return self.rep.array_to_circuit(self.circuit_padding[0][self.padding_len:self.padding_len+self.n_moments,:])

    def compute_step_reward(self):
        alpha=0.01
        circuit = self.get_qibo_circuit() #circuit.shape=(n_gates,5)
        print('Il circuito ricostruito e`: ',circuit.draw())
        #print('Circuit.shape= ',self.rep.circuit_to_array(circuit).shape) 
        true_circuit_label=self.labels
        learned_labels=np.asarray(circuit().state())
        mse = alpha*np.sqrt(np.abs(((true_circuit_label-learned_labels)**2).mean()))
        return -mse
            #... da continuare

    