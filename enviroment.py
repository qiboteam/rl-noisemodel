import gymnasium as gym
from gymnasium import spaces

class QuantumCircuitEnv(gym.Env):
    metadata = {"render_modes": None, "render_fps": None}

    def __init__(self, circuits):
        self.depth = circuits.shape[1] 
        self.observation_space = spaces.Dict({
          {
            'circuit': spaces.Box(),
            'position': 
            'noisy_channels'
          }
        })