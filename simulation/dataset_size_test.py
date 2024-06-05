from rlnoise.rl_agent import Agent
from rlnoise.gym_env import QuantumCircuit
import numpy as np

exp_folder = "simulation/experiments/test_size/"

config_file = exp_folder + "config.json"
dataset_file = exp_folder + "dataset.npz"
eval_dataset_file = exp_folder + "eval_dataset.npz"

result = []
size_list = [10, 20, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]
for size in size_list:
    env = QuantumCircuit(dataset_file = dataset_file, config_file = config_file)
    model_file_path = f"{exp_folder}/model_{size}"
    agent = Agent(config_file = config_file, env = env, model_file_path = model_file_path)
    size_result, _ = agent.apply_eval_dataset(eval_dataset_file, verbose=False)
    analize_result = np.array(
        [(
            size,
            size_result["fidelity"].mean(),
            size_result["mse"].mean(),
            size_result["trace_distance"].mean(),
            size_result["fidelity"].std(),
            size_result["mse"].std(),
            size_result["trace_distance"].std(),
        )], 
        dtype=[ 
            ('size','<i4'),
            ('fidelity_mean','<f4'),
            ('mse_mean','<f4'),
            ('trace_distance_mean','<f4'),
            ('fidelity_std','<f4'),
            ('mse_std','<f4'),
            ('trace_distance_std','<f4'),
        ])
    result.append(analize_result)
np.savez(f"{exp_folder}/size_test_result.npz", result=result)
for r in result:
    print(r)

