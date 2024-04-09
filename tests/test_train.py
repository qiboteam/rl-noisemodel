from rlnoise.rl_agent import Agent
from rlnoise.gym_env import QuantumCircuit

dataset_file = "tests/test_dataset.npz"
config_file = "tests/config_test.json"

env = QuantumCircuit(dataset_file = dataset_file, config_file = config_file)

agent = Agent(config_file = config_file, env = env)
agent.callback = None

agent.train(n_steps = 10000)