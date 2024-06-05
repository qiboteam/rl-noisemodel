from rlnoise.rl_agent import Agent
from rlnoise.gym_env import QuantumCircuit

exp_folder = "simulation/experiments/1q/"

config_file = exp_folder + "config.json"
dataset_file = exp_folder + "dataset.npz"

env = QuantumCircuit(dataset_file = dataset_file, config_file = config_file)

agent = Agent(config_file = config_file, env = env)
agent.train(n_steps = 100000)