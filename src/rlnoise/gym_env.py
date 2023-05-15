from configparser import ConfigParser
import numpy as np
import gym, random
from gym import spaces
import copy
from qibo.quantum_info import trace_distance

params=ConfigParser()
params.read("src/rlnoise/config.ini") 

neg_reward=params.getfloat('gym_env','neg_reward')
pos_reward=params.getfloat('gym_env','pos_reward')
step_r_metric=params.get('gym_env','step_r_metric')
action_penality=params.getfloat('gym_env','action_penality')
std_noise=params.getboolean('noise','std_noise')
coherent_noise=params.getboolean('noise','coherent_noise')
action_space_type=params.get('gym_env','action_space')
kernel_size = params.getint('gym_env','kernel_size')
step_reward=params.getboolean('gym_env','step_reward')

class QuantumCircuit(gym.Env):
    
    def __init__(self, circuits, labels, representation, reward, noise_param_space=None,
                 step_reward=step_reward, kernel_size=kernel_size, neg_reward=neg_reward,
                 pos_reward=pos_reward, step_r_metric=step_r_metric, action_penality=action_penality,
                 std_noise=std_noise, coherent_noise=coherent_noise, action_space_type=action_space_type):
        '''
        Args: 
            circuits (list): list of circuit represented as numpy vectors
            labels (list): list relative to the circuits
            representation: object of the class CircuitRepresentation()
            reward: object of the class DensityMatrixReward() or FrequencyReward()
            others: hyperparameters passed from config.ini
        '''
        super(QuantumCircuit, self).__init__(self)
        self.neg_reward=neg_reward
        self.pos_reward=pos_reward
        self.step_r_metric=step_r_metric
        self.action_penality=action_penality
        self.std_noise=std_noise
        self.coherent_noise=coherent_noise
        self.action_space_type=action_space_type
        self.kernel_size = kernel_size
        self.step_reward=step_reward 
        self.position=None
        self.circuits = circuits
        self.n_circ = len(self.circuits)
        self.n_qubits = circuits[0].shape[1]
        self.circuit_lenght = None
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
        self.current_state, self.current_target = self.init_state()
        #self.n_qubits = int(self.shape[-1] / self.rep.encoding_dim)
        self.actual_mse=None
        self.previous_mse=None
        if self.kernel_size is not None:
            assert self.kernel_size % 2 == 1, "Kernel_size must be odd"
            self.shape[2] = self.kernel_size
            self.kernel_size = self.kernel_size
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
        if self.action_space_type=="Box":
            self.action_space = spaces.Box( low=0, high=0.2,shape=(self.n_qubits,2), dtype=np.float32)

        elif self.action_space_type=="Binary":
            action_shape=[2,2,2,2,2,2]
            self.action_space = spaces.MultiDiscrete( 
            action_shape
            )
        
    def init_state(self, i=None):
        # initialize the state
        if i is None:
            i = random.randint(0, self.n_circ - 1)
        self.circuit_lenght=self.circuits[i].shape[0]
        state=copy.deepcopy(self.circuits[i])
        state=state.transpose(2,1,0) 
        self.shape = np.array(state.shape)
        assert (self.shape[0]) % (self.rep.encoding_dim) == 0
        if self.kernel_size is not None:
            padding = np.zeros(( self.shape[0],self.n_qubits, int(self.kernel_size/2)), dtype=np.float32)
            self.padded_circuit=np.concatenate((padding,state,padding), axis=2)

        return state, self.labels[i]
    
    def _get_obs(self):
        if self.kernel_size is not None:
            return self.get_kernel()
        else:
            return self.current_state #current_state.shape= (n_qubit, depth, encoding_dim )

    def _get_info(self):
        return { 'state': self._get_obs() } 

    def reset(self, i=None):
        self.position=0
        self.current_state, self.current_target = self.init_state(i)
        return self._get_obs()

    def step(self, action):
        if self.action_space_type =="Binary":
            action=action.reshape((self.n_qubits,2))
        reward=0.
        position = self.get_position()
        #print('\n \n current state BEFORE action: \n',self.current_state.transpose(1,2,0))
        if self.step_reward is True:
            if self.step_r_metric =="Trace_distance" or "trace_distance" or "td":
                self.previous_mse=trace_distance((self.current_target),(self.get_qibo_circuit()().state()))
            elif self.step_r_metric =='mse' or 'MSE':
                self.previous_mse=mse((self.current_target),(self.get_qibo_circuit()().state()))
       
        for q in range(self.n_qubits):          
            for idx,a in enumerate(action[q]):
                #print('considering qubit: ',q,'and action: ',action, 'at position: ',position)
                if self.std_noise is True:
                    if a!=0:
                        reward-=self.action_penality
                    if idx == 1:#to be generalized with channel2index
                        channel = self.noise_channels[idx]
                        self.current_state[self.rep.channel2index[channel],q, position]=a
                    if idx == 0:
                        channel = self.noise_channels[idx]
                        self.current_state[self.rep.channel2index[channel],q, position]=a
                if self.coherent_noise is True and self.std_noise is False:
                    if idx == 1:#to be generalized with channel2index
                        self.current_state[self.rep.epsilon2index["epsilon_z"],q, position]=a
                    if idx == 0:
                        self.current_state[self.rep.epsilon2index["epsilon_x"],q, position]=a                
                if self.step_reward:
                    reward+=self.step_reward_fun(a)

        if position == self.circuit_lenght - 1:
            # compute final reward
            terminated = True
        else:
            # update position
            self.position+=1
            terminated = False
        reward+=self.reward(self.get_qibo_circuit(), self.current_target, terminated)
        #print('current state AFTER action: \n',self.current_state.transpose(1,2,0))

        return self._get_obs(), reward, terminated, self._get_info()
    
    def step_reward_fun(self,action):
        '''
        Compute reward at each Agent step. 
        It will be positive if the action made has decreased the 
        distance, between predicted and real state, respect the distance at previous step, negative otherwise.

        Args:
            action: action performed by the agent, NOT USED YET

        Returns:
            step_reward
        '''
        #add penalization only if action !=0 (?)
        if self.step_r_metric=="Trace_distance" or "trace_distance" or "td":
            self.actual_mse=trace_distance((self.current_target),(self.get_qibo_circuit()().state()))
        elif self.step_r_metric=='mse' or 'MSE':
            self.actual_mse=mse((self.current_target),(self.get_qibo_circuit()().state()))
        if self.actual_mse>self.previous_mse:
            reward=self.neg_reward
        else:
            reward=self.pos_reward
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
        kernel.append(self.padded_circuit[:,:,pos:pos+self.kernel_size])

        return np.asarray(kernel,dtype=np.float32)
        
def mse(x,y):
    return np.sqrt(np.abs(((x-y)**2)).mean())



#Nota: aggiungere un elemento allazione in modo che l'agente possa scegliere anche di inserire una rotazione
#di epsilon_y, anche se non è presente nel coerente, per riuscire a ricostruire esattamente lo stato reale. (?)