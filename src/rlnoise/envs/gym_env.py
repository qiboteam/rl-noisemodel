import numpy as np
import gymnasium as gym
from gymnasium import spaces
from qibo import gates
from qibo.models import Circuit
from copy import deepcopy
from rlnoise.utils import truncated_moments_matching
from rlnoise.rewards.observables_reward import obs_reward
from rlnoise.rewards.density_matrix_reward import dm_reward


class CircuitsGym(gym.Env):

    def __init__(self, circuits_repr, labels, reward_func=truncated_moments_matching, reward_method="dm"):

        self.actions=(0,1)
        self.labels=labels
        self.len = len(circuits_repr[0])
        self.shape = np.shape(circuits_repr[0])
        self.circuits_repr = circuits_repr
        self.set_reward_func(reward_func=reward_func)
        self.set_reward_method(reward_method=reward_method)

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

    def get_reward_method(self):
        return self.reward_method

    def set_reward_func(self, reward_func):
        self.reward_func=reward_func  

    def set_reward_method(self, reward_method):
        self.reward_method=reward_method

    def reset(self, seed=42, verbose=False, sample=None):

        super().reset(seed=seed)
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
            if self.reward_method=="observables":
                reward = obs_reward(
                    circuit=self.circuits_repr[self.sample],
                    noisy_channels=self.noisy_channels,
                    label=self.labels[self.sample], 
                    reward_func=self.reward_func
                    )
            elif self.reward_method=="dm":
                reward = dm_reward( 
                    circuit=self.circuits_repr[self.sample],
                    noisy_channels=self.noisy_channels,
                    label=self.labels[self.sample]
                    )
            else:
                print("Use a defined reward method")
            observation=self._get_obs()
            if verbose:
                self._get_info(last_step=True)
                print("REWARD: ", reward)
            return observation, reward, True
        else:
            reward=0
            self.position+=1
            observation=self._get_obs()
            if verbose:
                self._get_info()
            return observation, reward, False
    
    def _get_obs(self):

        self.observation_space[:,3].fill(0.)
        self.observation_space[self.position,3] = 1.
        self.observation_space[:,2] = self.noisy_channels
        return self.observation_space.copy()

    def _get_info(self, last_step=False):
        print("Circuit number: ", self.sample)
        if last_step:
            print("EPISODE ENDED")
        else:
            print("Action number: ", self.position)
        print("Last action: ", self.last_action)
        print("Observation:")
        print(self._get_obs())