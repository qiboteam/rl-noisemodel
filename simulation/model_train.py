from rlnoise.rl_agent import Agent
from rlnoise.gym_env import QuantumCircuit

dataset_file = "simulation/experiments/3q_high_noise/dataset.npz"
config_file = "simulation/experiments/3q_high_noise/config.json"

env = QuantumCircuit(dataset_file = dataset_file, config_file = config_file)

agent = Agent(config_file = config_file, env = env)
agent.train(n_steps = 400000)