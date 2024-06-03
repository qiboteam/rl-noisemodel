from rlnoise.rl_agent import Agent
from rlnoise.gym_env import QuantumCircuit

exp_folder = "simulation/experiments/test_size/"

config_file = exp_folder + "config.json"
dataset_file = exp_folder + "dataset.npz"

size_list = [10, 20, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]
for size in size_list:
    env = QuantumCircuit(dataset_file = dataset_file, config_file = config_file, reduced_size = size)
    agent = Agent(config_file = config_file, env = env)
    agent.callback.save_path = f"{exp_folder}/model_{size}"
    agent.train(n_steps = 2000000)