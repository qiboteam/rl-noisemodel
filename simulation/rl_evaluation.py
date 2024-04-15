from rlnoise.rl_agent import Agent
from rlnoise.gym_env import QuantumCircuit
import numpy as np

model_file_path = "simulation/experiments/3q_high_noise/model.zip"
dataset_file = "simulation/experiments/3q_high_noise/dataset.npz"
config_file = "simulation/experiments/3q_high_noise/config.json"
rb_dataset = "simulation/experiments/3q_high_noise/rb_dataset.npz"
result_file = "simulation/experiments/3q_high_noise/rl_eval.npz"

env = QuantumCircuit(dataset_file = dataset_file, config_file = config_file)

agent = Agent(config_file = config_file, env = env)
result = agent.apply_rb_dataset(rb_dataset)

print("Result of RL evaluation, the columns are: depth, fidelity, fidelity_std, trace_distance, trace_distance_std.")
print(result)

np.savez(result_file, result=result)