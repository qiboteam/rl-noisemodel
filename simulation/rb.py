from rlnoise.randomized_benchmarking import run_rb, rb_evaluation
import numpy as np
from rlnoise.rl_agent import Agent
from rlnoise.gym_env import QuantumCircuit

exp_folder = "simulation/experiments/3q_low/"

config_file = exp_folder + "config.json"
rb_dataset = exp_folder + "rb_dataset.npz"
result_file_rb = exp_folder + "rb_result.npz"
result_file_rl = exp_folder + "rl_result.npz"
model_file_path = exp_folder + "model.zip"
dataset_file = exp_folder + "dataset.npz"
config_file = exp_folder + "config.json"

optimal_params = run_rb(rb_dataset, config_file)
print("RB Model:")
print(optimal_params)
decay_constant = 1 - optimal_params["l"]    
print("Decay constant: ", decay_constant)

result = rb_evaluation(decay_constant, rb_dataset, config_file)

np.savez(result_file_rb, result=result)

env = QuantumCircuit(dataset_file = dataset_file, config_file = config_file)

agent = Agent(config_file = config_file, env = env, model_file_path = model_file_path)
result = agent.apply_rb_dataset(rb_dataset)

np.savez(result_file_rl, result=result)

