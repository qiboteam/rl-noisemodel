import numpy as np
import qibo
from rlnoise.rl_agent import Agent
from rlnoise.gym_env import QuantumCircuit

qibo.set_backend("qibojit",platform="numba")

#exp_folder = "simulation/experiments/3q_large/"

exp_folder = "hardware/experiments/qw11q/"

config_file = exp_folder + "config_qibolab.json"
model_file_path = exp_folder + "model.zip"
dataset_file = exp_folder + "dataset.npz"
eval_dataset_file = exp_folder + "eval_dataset.npz"
result_file_rl = exp_folder + "evaluation_result.npz"


env = QuantumCircuit(dataset_file = dataset_file, config_file = config_file)

agent = Agent(config_file = config_file, env = env, model_file_path = model_file_path)
result, dms = agent.apply_eval_dataset(eval_dataset_file)

np.savez(result_file_rl, result=result, dms=dms)