from rlnoise.rl_agent import Agent
from rlnoise.gym_env import QuantumCircuit

dataset_file = "simulation/experiments/3q_high_noise/dataset.npz"
config_file = "simulation/experiments/3q_high_noise/config.json"
model_file = "simulation/experiments/3q_high_noise/model"

env = QuantumCircuit(dataset_file = dataset_file, config_file = config_file)

agent = Agent(config_file = config_file, env = env, model_file_path = model_file)
agent.train(n_steps = 50000)