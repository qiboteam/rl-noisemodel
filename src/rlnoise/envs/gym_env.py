import numpy as np
import gym
from gym import spaces
from qibo import gates
from qibo.models import Circuit
from copy import deepcopy
from utils import truncated_moments_matching


class CircuitsGym(gym.Env):

    def __init__(self, circuits_repr, labels, reward_func=truncated_moments_matching):
        super(CircuitsGym, self).__init__()

        self.actions=(0,1)
        self.labels=labels
        self.len = len(circuits_repr[0])
        self.shape = np.shape(circuits_repr[0])
        self.circuits_repr = circuits_repr
        self.set_reward_func(reward_func=reward_func)

        self.observation_space = spaces.Box(low=0, high=1, shape=(self.len, 4), dtype=float)
        self.action_space = spaces.Discrete(2)

    def n_elements(self):
        return len(self.labels)

    def get_circuits_repr(self):
        return self.circuits_repr

    def get_labels(self):
        return self.labels

    def get_reward_func(self):
        return self.reward_func

    def reset(self, verbose=False, sample=None):

        #super().reset(seed=seed)
        self.position = 0
        self.last_action = None
        if sample==None:
            self.sample = np.random.randint(low=0, high=len(self.circuits_repr))
        else:
            self.sample=sample
        self.circuit = self.circuits_repr[self.sample]
        self.noisy_channels = np.zeros((self.len))
        self.observation_space = np.zeros((self.len, 4), dtype=np.float32)
        self.observation_space[:,0:2] = self.circuit
        if verbose:
            print("EPISODE STARTED")
            self._get_info()
        return self._get_obs()

    def get_sample(self):
        return self.sample

    def step(self, action, verbose=False):
        self.last_action=action
        if action == 1:
            self.noisy_channels[self.position]=1.
        # Check for termination.
        if self.position == (self.len-1):
            # Compute reward
            reward = self.compute_reward(self.labels[self.sample], n_shots=100)
            observation=self._get_obs()
            if verbose:
                self._get_info(last_step=True)
                print("REWARD: ", reward)
            return observation, reward, True, self._get_info()
        else:
            reward=0
            self.position+=1
            observation=self._get_obs()
            if verbose:
                self._get_info()
            return observation, reward, False, self._get_info()
    
    def _get_obs(self):

        self.observation_space[:,3].fill(0.)
        self.observation_space[self.position,3] = 1.
        self.observation_space[:,2] = self.noisy_channels
        return self.observation_space.copy()

    def _get_info(self, last_step=False):
        #print("Circuit number: ", self.sample)
        if last_step:
            print("EPISODE ENDED")
        else:
            pass
            #print("Action number: ", self.position)
        print("Last action: ", self.last_action)
        #print("Observation:")
        info = {'episode': self.sample, 'observation': self._get_obs()}
        #print(info['observation'])
        return info
        
    """---------------------------------------"""
    """--------COMPUTE REWARD-----------------"""
    """---------------------------------------"""

    def set_reward_func(self, reward_func):
        self.reward_func=reward_func  

    def compute_reward(self, label, n_shots=100):
        reward=0.
        generated_circuit = self.generate_circuit()
        observables = np.ndarray((3,2), dtype=float)
        index=0
        for obs in ["Z", "Y", "X"]:
            moments=self.pauli_probabilities(generated_circuit, obs, n_shots=n_shots)
            observables[index, :]=moments
            index+=1
        for i in range(3):
            reward+=self.reward_func(m1=observables[i,0], m2=label[i,0], v1=observables[i,1], v2=label[i,1])
        return reward

    def generate_circuit(self, dep_error=0.05):
      qibo_circuit = Circuit(1, density_matrix=True)
      for i in range(self.len):
        if self.circuit[i,0]==0:
          qibo_circuit.add(gates.RZ(0, theta=self.circuit[i,1]*2*np.pi, trainable=False))
        else:
          qibo_circuit.add(gates.RX(0, theta=self.circuit[i,1]*2*np.pi, trainable=False))
        if self.noisy_channels[i]==1:
          qibo_circuit.add(gates.DepolarizingChannel((0,), lam=dep_error))
      return qibo_circuit

    def pauli_probabilities(self, circuit, observable, n_shots=100, n_rounds=100):
        measured_circuit = deepcopy(circuit)
        self.add_masurement_gates(measured_circuit, observable=observable)
        register=np.ndarray((n_rounds,), dtype=float)
        moments=np.ndarray((2,), dtype=float)
        for i in range(n_rounds):
            probs=self.compute_shots(measured_circuit, n_shots=n_shots)
            register[i]=probs[0]-probs[1]
        moments[0]=np.mean(register)
        moments[1]=np.var(register)
        return moments

    def add_masurement_gates(self, circuit, observable):
        if observable=='X' or observable=='Y':
            circuit.add(gates.H(0))
        if observable=='Y':
            circuit.add(gates.SDG(0))
        circuit.add(gates.M(0))
        
    def compute_shots(self, circuit, n_shots):
        shots_register_raw = circuit(nshots=n_shots).frequencies(binary=False)
        shots_register=tuple(int(shots_register_raw[key]) for key in range(2))
        return np.asarray(shots_register, dtype=float)/float(n_shots)
