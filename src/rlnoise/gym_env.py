import numpy as np
import gym, random
from gym import spaces

# currently working just for single qubit circuits
# TO DO:
# - Adapt to multi-qubits circuits
# - Implement batches
# - Write a reward that makes sense
# - Adapt it for working with the 3d representation

NEG_REWARD=-0.1
POS_REWARD=0.1
class QuantumCircuit(gym.Env):
    
    def __init__(self, circuits, representation, labels, reward, noise_param_space=None, kernel_size=None,step_reward=True):
        '''
        Args: 
            circuits (list): list of circuit represented as numpy vectors
        
        '''
        super(QuantumCircuit, self).__init__()
        self.position=None
        self.circuits = circuits
        self.n_circ = len(self.circuits)
        self.n_qubits = circuits[0].shape[1]
        self.circuit_lenght = None
        self.rep = representation
        self.n_gate_types = len(self.rep.gate2index)
        self.n_channel_types = len(self.rep.channel2index)
        self.noise_channels = list(self.rep.channel2index.keys())
        self.kernel_size = kernel_size
        self.step_reward=step_reward
        if noise_param_space is None:
            self.noise_par_space = { 'range': (0,0.1), 'n_steps': 100 } # convert this to a list of dict for allowing custom range and steps for the different noise channels
        else:
            self.noise_par_space = noise_param_space
        self.noise_incr = np.diff(self.noise_par_space['range']) / self.noise_par_space['n_steps']

        assert self.noise_incr > 0
        self.labels = labels
        self.reward = reward
        self.current_state, self.current_target = self.init_state()
        #self.n_qubits = int(self.shape[-1] / self.rep.encoding_dim)
        self.actual_mse=None
        self.previous_mse=None
        if kernel_size is not None:
            assert kernel_size % 2 == 1, "Kernel_size must be odd"
            self.shape[2] = kernel_size
            self.kernel_size = kernel_size
        else:
            self.kernel_size = None
            self.shape[0] += 1
        #print('self.shape2: ',self.shape)
        self.observation_space = spaces.Box(
            low = 0,
            high = 1,
            shape = tuple(self.shape),
            dtype = np.float32
        )
        #action_shape=([ self.n_channel_types + 1 for i in range(self.n_qubits) ] +  [ self.noise_par_space['n_steps'] for i in range(self.n_qubits) ])
        '''
        self.action_space = spaces.MultiDiscrete(
            [self.n_qubits, 2]     # +1 for the no ch.annel option 
            
        )
        self.action_space = spaces.Box( low=0., high=1,shape=(self.n_qubits,2), dtype=np.float32)
        '''
        self.action_space = spaces.Box( low=0., high=0.2,shape=(self.n_qubits,2), dtype=np.float32)
        # doesn't work with stable baselines
        #self.action_space = spaces.Dict({
        #    "channel": spaces.MultiDiscrete([ self.n_channel_types for i in range(self.n_qubits) ]),
        #    "param": spaces.Box(low=0, high=1, shape=(self.n_qubits,), dtype=np.float32)
        #})
        
    def init_state(self, i=None):
        # initialize the state
        if i is None:
            i = random.randint(0, self.n_circ - 1)
        self.circuit_lenght=self.circuits[i].shape[0]
        #print('circuit shape: ',self.circuits[i].shape)
        state=self.circuits[i]
        state=state.transpose(2,1,0) #rearranged in shape (1, num_qubits, depth, encoding_dim+1)
        #print('state shape: ',state.shape)
        self.shape = np.array(state.shape)
        #print('self.shape: ',self.shape)
        assert (self.shape[0]) % (self.rep.encoding_dim) == 0

        if self.kernel_size is not None:
            padding = np.zeros(( self.shape[0],self.n_qubits, int(self.kernel_size/2)), dtype=np.float32)
            self.padded_circuit=np.concatenate((padding,state,padding), axis=2)
            #print('padding shape: ',self.padded_circuit.shape)
            
        return state, self.labels[i]
    
    def _get_obs(self):
        if self.kernel_size is not None:
            return self.get_kernel()
        else:
            return self.current_state #current_state.shape= (1, depth, 6 if only 1qubit )

    def _get_info(self):
        return { 'state': self._get_obs() } 

    def reset(self, i=None):
        self.position=0
        self.current_state, self.current_target = self.init_state(i)
        #print('current state shape: ', self.current_state.shape)
        return self._get_obs()

    def step(self, action):
        reward=0.
        position = self.get_position()
        std_noise=True
        coherent_noise=False
        #print('current state BEFORE action: \n',action,'\n',np.round((self.current_state.transpose(1,2,0)),decimals=4))
        if self.step_reward:
            self.previous_mse=mse((self.current_target),(self.get_qibo_circuit()().state()))
        #print('> State:\n', self._get_obs())
        
        for q in range(len(action)):
            #print('action: ',action)
            for idx,a in enumerate(action[q]):

               # print('considering qubit: ',q,'and action: ',action, a, 'at position: ',position)
                if a !=0  : #add gate only if action > of something
                    reward-=0.01
                    if std_noise is True:
                        if idx == 1:#to be generalized with channel2index
                            channel = self.noise_channels[idx](q,t1=1,t2=1, time=a) # -1 cause there is no identity channel in self.noise_channels
                        if idx == 0:
                            channel = self.noise_channels[idx](q,lam=a)

                        #self.current_state[:,q, position] += self.rep.gate_to_array(channel)
                        self.current_state[self.rep.channel2index[type(channel)],q, position]=a
                    if coherent_noise is True and std_noise is False:
                        if idx == 1:#to be generalized with channel2index
                            #gate = gates.RX(q,theta=a) # -1 cause there is no identity channel in self.noise_channels
                            self.current_state[-1,q, position]=a
                           
                        if idx == 0:
                            #gate = gates.RZ(q,theta=a)    
                            self.current_state[-2,q, position]=a                    

                    if self.step_reward:
                        reward+=self.step_reward_fun()

                #self.current_state[0, position, idx:idx+self.rep.encoding_dim] += self.rep.gate_to_array(channel)
        #print(f'> New State: \n{self._get_obs()}')
        if position == self.circuit_lenght - 1:
            # compute final reward
            terminated = True
        else:
            # update position
            self.position+=1
            terminated = False
        reward+=self.reward(self.get_qibo_circuit(), self.current_target, terminated)
        #print('current state AFTER action: \n',np.round((self.current_state.transpose(1,2,0)),decimals=4))
        #print('current state shape: ',self.current_state.transpose(1,2,0).shape)
        return self._get_obs(), reward, terminated, self._get_info()
    
    def step_reward_fun(self):
        
        self.actual_mse=mse((self.current_target),(self.get_qibo_circuit()().state()))
        if self.actual_mse>self.previous_mse:
            reward=NEG_REWARD
        else:
            reward=POS_REWARD
        return reward
    def render(self):
        print(self.get_qibo_circuit().draw(), end='\r')
            
    def get_position(self):
        #return (self.current_state[:,0,-1] == 1).nonzero()[-1]
        return self.position

    def get_qibo_circuit(self):
        return self.rep.rep_to_circuit(self.current_state.transpose(2,1,0)[:,:,:])
    
    def get_circuit_rep(self):
        return self.current_state.transpose(1,2,0)

    def get_kernel(self):
        pos = int(self.get_position())
        kernel=[]
        r=int(self.kernel_size/2)
        self.padded_circuit[:,:,r:-r]=self.current_state
        #print(self.padded_circuit)
         
        kernel.append(self.padded_circuit[:,:,pos:pos+self.kernel_size])
        
        return np.asarray(kernel,dtype='float32')
        
def mse(x,y):
    return np.sqrt(np.abs(((x-y)**2)).mean())
#Riscrivere rappresentazione in modo che solo 2 colonne rappresentano i channel: se diverse da zero indicano direttamente il valore del parametro del canale (time o lambda)
#eliminare colonna della posizione