import numpy as np
import qibo
from rlnoise.rl_agent import Agent
from rlnoise.gym_env import QuantumCircuit

qibo.set_backend("qibojit",platform="numba")

estimate_dm_noise = True

exp_folder = "experiments/hardware/qw11qD4/"

config_file = exp_folder + "config.json"
model_file_path = exp_folder + "model.zip"
dataset_file = exp_folder + "dataset.npz"
eval_dataset_file = exp_folder + "eval_dataset.npz"
result_file_rl = exp_folder + "evaluation_result.npz"


env = QuantumCircuit(dataset_file = dataset_file, config_file = config_file)

agent = Agent(config_file = config_file, env = env, model_file_path = model_file_path)
result, dms_rl = agent.apply_eval_dataset(eval_dataset_file)

if estimate_dm_noise:
    from rlnoise.utils import estimate_hardware_noise
    dataset = np.load(eval_dataset_file, allow_pickle=True)
    dms_true = dataset['labels']
    tot_error = 0.
    for i in range(len(dms_true)):
        dm_rl, dm_true = dms_rl[i], dms_true[i]
        tot_error += estimate_hardware_noise(dms_rl=dm_rl, dms_true=dm_true)
    print("Avg Eroor: ", tot_error/len(dms_true))

np.savez(result_file_rl, result=result, dms=dms_rl)