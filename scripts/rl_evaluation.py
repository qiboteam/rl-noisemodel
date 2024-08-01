import numpy as np
import qibo
from rlnoise.rl_agent import Agent
from rlnoise.gym_env import QuantumCircuit

qibo.set_backend("qibojit",platform="numba")

estimate_hardware_noise = False

exp_folder = "experiments/simulation/1q/"

config_file = exp_folder + "config.json"
model_file_path = exp_folder + "model.zip"
dataset_file = exp_folder + "dataset.npz"
eval_dataset_file = exp_folder + "eval_dataset.npz"
result_file_rl = exp_folder + "evaluation_result.npz"


env = QuantumCircuit(dataset_file = dataset_file, config_file = config_file)

agent = Agent(config_file = config_file, env = env, model_file_path = model_file_path)
result, dms_rl = agent.apply_eval_dataset(eval_dataset_file)

if estimate_hardware_noise:
    from rlnoise.utils import estimate_hardware_noise
    dataset = np.load(dataset_file, allow_pickle=True)
    dms_true = dataset['dms']
    estimate_hardware_noise(dms_rl, dms_true)

np.savez(result_file_rl, result=result, dms=dms_rl)