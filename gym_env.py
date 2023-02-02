import gymnasium as gym
from gymnasium import spaces
import numpy as np

class QuantumCircuitEnv(gym.Env):
  '''Create gym environment to represent quantum circuit and all possible actions'''

  metadata = {"render_modes": None, "render_fps": None}

  def __init__(self, circuit):
    self.len=circuit.shape[0]
    self.depth = circuit.shape[1] 
    self.circuit_shape = circuit.shape
    self.observation_space = spaces.Dict({
      {
        'circuit': spaces.Box(low=0, high=1, shape=self.circuit_shape, dtype=np.float),
        'position': spaces.Discrete(self.len),
        'noisy_channels': spaces.Box(low=0, high=1, shape=self.len, dtype=np.float)
      }
    })
    self.action_space = spaces.Discrete(2)
    self._action_to_

  def _get_obs(self):
    return 

  def step(self, action):
    self.position+=1