from rlnoise.rl_agent import Agent
from rlnoise.gym_env import QuantumCircuit
import numpy as np

exp_folder = "simulation/experiments/3q_multiple/"

config_file = exp_folder + "config.json"
dataset_file = exp_folder + "dataset.npz"
eval_dataset_file = exp_folder + "eval_dataset.npz"
result = []
model_list = [i for i in range(1,9)]
for model in model_list:
    env = QuantumCircuit(dataset_file = dataset_file, config_file = config_file)
    model_file_path = f"{exp_folder}/model_{model}"
    agent = Agent(config_file = config_file, env = env, model_file_path = model_file_path)
    model_result, _ = agent.apply_eval_dataset(eval_dataset_file, verbose=False)
    analize_result = np.array(
        [(
            model,
            model_result["fidelity"].mean(),
            model_result["mse"].mean(),
            model_result["trace_distance"].mean(),
            model_result["fidelity"].std(),
            model_result["mse"].std(),
            model_result["trace_distance"].std(),
        )], 
        dtype=[ 
            ('model','<i4'),
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

