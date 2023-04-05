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
    
    def __init__(self, circuits, representation, labels, reward, noise_param_space=None, kernel_size=None):
        super(QuantumCircuit, self).__init__()

        self.circuits = circuits[:,np.newaxis,:,:]
        self.n_circ = self.circuits.shape[0]
        self.n_moments = self.circuits.shape[2]
        self.rep = representation
        self.n_gate_types = len(self.rep.gate2index)
        self.n_channel_types = len(self.rep.channel2index)
        self.noise_channels = list(self.rep.channel2index.keys())
        if noise_param_space is None:
            self.noise_par_space = { 'range': (0,0.1), 'n_steps': 100 } # convert this to a list of dict for allowing custom range and steps for the different noise channels
        else:
            self.noise_par_space = noise_param_space
        self.noise_incr = np.diff(self.noise_par_space['range']) / self.noise_par_space['n_steps']
        assert self.noise_incr > 0
        self.labels = labels
        self.reward = reward

        shape = list(self.circuits.shape[1:])
        assert shape[-1] % (self.rep.encoding_dim) == 0
        self.n_qubits = int(shape[-1] / self.rep.encoding_dim)
        if kernel_size is not None:
            assert kernel_size % 2 == 1, "Kernel_size must be odd"
            shape[0] = kernel_size
            self.kernel_size = kernel_size
        else:
            self.kernel_size = None
            shape[-1] += 1
        
        self.observation_space = spaces.Box(
            low = 0,
            high = 1,
            shape = tuple(shape),
            dtype = np.float32
        )

        self.action_space = spaces.MultiDiscrete(
            [ self.n_channel_types + 1 for i in range(self.n_qubits) ] +       # +1 for the no channel option 
            [ self.noise_par_space['n_steps'] for i in range(self.n_qubits) ]
        )
        # doesn't work with stable baselines
        #self.action_space = spaces.Dict({
        #    "channel": spaces.MultiDiscrete([ self.n_channel_types for i in range(self.n_qubits) ]),
        #    "param": spaces.Box(low=0, high=1, shape=(self.n_qubits,), dtype=np.float32)
        #})
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
        if self.kernel_size is not None:
            return self.get_kernel()
        else:
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
            if a[0] != 0:
                idx = q * self.rep.encoding_dim
                lam = self.noise_par_space['range'][0] + a[1] * self.noise_incr
                channel = self.noise_channels[a[0]-1](q, lam=lam) # -1 cause there is no identity channel in self.noise_channels
                self.current_state[0, position, idx:idx+self.rep.encoding_dim] += self.rep.gate_to_array(channel)
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

    def get_kernel(self):
        pos = self.get_position()
        l = len(self.current_state[0])
        r = int(self.kernel_size / 2)
        if pos - r < 0:
            pad = r - pos + 1
            tmp = np.pad(
                self.current_state,
                pad_width = ((0,0), (0,0), (pad, 0)),
                mode = 'constant',
                constant_values = 0
            )
        elif pos + r >= l:
            pad = pos + r - l + 1
            tmp = np.pad(
                self.current_state,
                pad_width = ((0,0), (0,0), (0, pad)),
                mode = 'constant',
                constant_values = 0
            )
        else:
            tmp = self.current_state
        
        kernel = tmp[:,pos-r:pos+r+1,:-1]
        return kernel
        


    
