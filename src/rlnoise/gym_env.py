from configparser import ConfigParser
import numpy as np
import gym, random
from gym import spaces
import copy
from qibo import gates
from qibo.quantum_info import trace_distance
from rlnoise.dataset import gate_to_idx

def gate_action_index(gate):
    if gate == 'epsilon_x':
        return 0
    if gate == 'epsilon_z':
        return 1
    if gate == gates.ResetChannel:
        return 2
    if gate == gates.DepolarizingChannel:
        return 3

class QuantumCircuit(gym.Env):
    
    def __init__(self, circuits, labels, representation, reward, noise_param_space=None,
                 step_reward=None, kernel_size=None, neg_reward=None,pos_reward=None, 
                 step_r_metric=None, action_penality=None,
                 action_space_type=None):
        '''
        Args: 
            circuits (list): list of circuit represented as numpy vectors
            labels (list): list relative to the circuits
            representation: object of the class CircuitRepresentation()
            reward: object of the class DensityMatrixReward() or FrequencyReward()
            others: hyperparameters passed from config.ini
        '''
        super(QuantumCircuit, self).__init__()
        self.neg_reward=neg_reward
        self.pos_reward=pos_reward
        self.step_r_metric=step_r_metric
        self.action_penality=action_penality
        self.action_space_type=action_space_type
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 == 1, "Kernel_size must be odd"
        self.step_reward=step_reward 
        self.position=None
        self.circuits = circuits
        self.n_circ = len(self.circuits)
        self.n_qubits = circuits[0].shape[1]
        self.circuit_lenght = None
        self.rep = representation
        self.actual_mse=None
        self.previous_mse=None
        self.labels = labels
        self.reward = reward
        self.encoding_dim = 8
        self.observation_space = spaces.Box(
            low = 0,
            high = 1,
            shape = (self.encoding_dim,self.n_qubits,self.kernel_size),
            dtype = np.float32
        )
        if self.action_space_type=="Continuous":
            self.action_space = spaces.Box( low=0, high=0.2,shape=(self.n_qubits,4), dtype=np.float32)

        elif self.action_space_type=="Discrete":
            if noise_param_space is None:
                self.noise_par_space = { 'range': (0,0.1), 'n_steps': 100 }
            else:
                self.noise_par_space = noise_param_space
            self.noise_incr = np.diff(self.noise_par_space['range']) / self.noise_par_space['n_steps']
            assert self.noise_incr > 0

        elif self.action_space_type=="Binary":
            action_shape=[2 for _ in range(self.n_qubits*4)]
            self.action_space = spaces.MultiDiscrete( 
            action_shape
            )
        
    def init_state(self, i=None):
        if i is None:
            i = random.randint(0, self.n_circ - 1)
        self.circuit_number=i
        self.circuit_lenght=self.circuits[i].shape[0]
        state=copy.deepcopy(self.circuits[i])
        state=state.transpose(2,1,0) #check what returns from circuit_to_array and use proper shape
        padding = np.zeros(( self.encoding_dim,self.n_qubits, int(self.kernel_size/2)), dtype=np.float32)
        self.padded_circuit=np.concatenate((padding,state,padding), axis=2)
        return state, self.labels[i]
    
    def _get_obs(self):
        pos = int(self.get_position())
        kernel=[]
        r=int(self.kernel_size/2)
        self.padded_circuit[:,:,r:-r]=self.current_state
        kernel.append(self.padded_circuit[:,:,pos:pos+self.kernel_size])
        return np.asarray(kernel,dtype=np.float32)

    def _get_info(self):
        #Add last action
        return {'State': self._get_obs(),
                'Pos': self.position,
                'Circ': self.circuit_number} 

    def reset(self, i=None):
        self.position=0
        self.current_state, self.current_target = self.init_state(i)
        return self._get_obs()

    def step(self, action):
        if self.action_space_type =="Binary" or self.action_space_type =="Discrete":
            action=action.reshape((self.n_qubits,4))
        reward=0.
        position = self.get_position()
        #print('\n \n current state BEFORE action: \n',self.current_state.transpose(1,2,0))
        if self.step_reward is True:
            if self.step_r_metric.lower() == "trace_distance":
                self.previous_mse=trace_distance((self.current_target),(self.get_qibo_circuit()().state()))
            elif self.step_r_metric.lower =='mse':
                self.previous_mse=mse((self.current_target),(self.get_qibo_circuit()().state()))
            else:
                raise("Error")
       
        for q in range(self.n_qubits):          
            for idx, a in enumerate(action[q]):
                #print('considering qubit: ',q,'and action: ',action, 'at position: ',position)
                if a!=0:
                    reward -= self.action_penality
                    if idx == gate_action_index("epsilon_x"):
                        self.current_state[gate_to_idx("epsilon_x"), q, position]=a
                    if idx == gate_action_index("epsilon_z"):
                        self.current_state[gate_to_idx("epsilon_z"),q, position]=a
                    if idx == gate_action_index(gates.ResetChannel):
                        self.current_state[gate_to_idx(gates.ResetChannel),q, position]=a
                    if idx == gate_action_index(gates.DepolarizingChannel):
                        self.current_state[gate_to_idx(gates.DepolarizingChannel),q, position]=a                
                    if self.step_reward:
                        reward+=self.step_reward_fun(a)
        if position == self.circuit_lenght - 1:
            terminated = True
        else:
            self.position+=1
            terminated = False
        reward+=self.reward(self.get_qibo_circuit(), self.current_target, terminated)
        #print('current state AFTER action: \n',self.current_state.transpose(1,2,0))
        return self._get_obs(), reward, terminated, self._get_info()
    
    def step_reward_fun(self, action):
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
        return self.position

    def get_qibo_circuit(self):
        return self.rep.rep_to_circuit(self.current_state.transpose(2,1,0)[:,:,:])
    
    def get_circuit_rep(self):
        return self.current_state.transpose(1,2,0)

def mse(x,y):
    return np.sqrt(np.abs(((x-y)**2)).mean())
