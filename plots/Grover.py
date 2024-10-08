from rlnoise.noise_model import CustomNoiseModel
from rlnoise.randomized_benchmarking import fill_identity
from rlnoise.rl_agent import Agent
from rlnoise.gym_env import QuantumCircuit
from rlnoise.utils import grover, mms, mse, compute_fidelity
from qibo.noise import NoiseModel, DepolarizingError
import json
import numpy as np

exp_folder = "experiments/simulation/3q_low/"
model_file = exp_folder + "model.zip"
config_file = exp_folder + "config.json"
dataset_file = exp_folder + "dataset.npz"
lambda_rb = 0.0156

circuit = grover()
noise_model = CustomNoiseModel(config_file=config_file)
noisy_circuit = noise_model.apply(grover())

env = QuantumCircuit(dataset_file = dataset_file, config_file = config_file)
rl_noise_model = Agent(config_file = config_file, env = env, model_file_path = model_file)
rl_noisy_circuit = rl_noise_model.apply(grover())

noise = NoiseModel()
noise.add(DepolarizingError(lambda_rb))
rb_processed_circuit = fill_identity(grover())
RB_noisy_circuit = noise.apply(rb_processed_circuit)

dm_truth = noisy_circuit().state()
dm_rl = rl_noisy_circuit().state()
dm_RB = RB_noisy_circuit().state()

print("Circuit Info:")
print("Length: ", len(circuit.queue))
print("Moments: ", len(circuit.queue.moments))
print("Fidelity:")
print("No noise: ", compute_fidelity(dm_truth, circuit().state()))
print("RL agent: ", compute_fidelity(dm_truth, dm_rl))
print("RB noise: ", compute_fidelity(dm_truth, dm_RB))
print("MMS: ", compute_fidelity(dm_truth, mms(8)))
print("MSE:")
print("No noise: ", mse(dm_truth, circuit().state()))
print("RL agent: ", mse(dm_truth, dm_rl))
print("RB noise: ", mse(dm_truth, dm_RB))
print("MMS: ", mse(dm_truth, mms(8)))

result_dict = {}
result_dict["fidelity"] = {}
result_dict["mse"] = {}
result_dict["fidelity"]["no_noise"] = float(compute_fidelity(dm_truth, circuit().state()))
result_dict["fidelity"]["RL"] = float(compute_fidelity(dm_truth, dm_rl))
result_dict["fidelity"]["RB"] = float(compute_fidelity(dm_truth, dm_RB))
result_dict["fidelity"]["MMS"] = float(compute_fidelity(dm_truth, mms(8)))
result_dict["mse"]["no_noise"] = float(mse(dm_truth, circuit().state()))
result_dict["mse"]["RL"] = float(mse(dm_truth, dm_rl))
result_dict["mse"]["RB"] = float(mse(dm_truth, dm_RB))
result_dict["mse"]["MMS"] = float(mse(dm_truth, mms(8)))

# Save result to json file
with open(exp_folder + "images/Grover_result.json", "w") as f:
    json.dump(result_dict, f)

def compute_probabilities(rho):
    probs = {}
    for i in range(8):
        probs[format(i, "03b")] = np.abs(rho[i, i])
    return probs
            
no_noise_shots = compute_probabilities(circuit().state())
noise_shots = compute_probabilities(dm_truth)
rl_shots = compute_probabilities(dm_rl)
RB_shots = compute_probabilities(dm_RB)

print("Probs:")
print("No noise", no_noise_shots)
print("Noise", noise_shots)
print("RL", rl_shots)
print("RB", RB_shots)

SMALL_SIZE = 22
MEDIUM_SIZE = 26
BIGGER_SIZE = 28

import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# Extract keys and values from dictionaries
keys = list(no_noise_shots.keys())
values1 = list(noise_shots.values())
values2 = list(rl_shots.values())
values3 = list(RB_shots.values())
# Set the width of the bars
bar_width = 0.2
# Set the positions of bars on X-axis
r1 = range(len(keys))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

fig=plt.figure(figsize=(12, 7))
ax=fig.add_subplot(111)
ax.bar(r1, values1, width=bar_width, label='Ground truth', color='#e60049')
ax.bar(r2, values2, width=bar_width, label='RL', color='#0bb4ff')
ax.bar(r3, values3, width=bar_width, label='RB', color='green')

plt.xlabel('State')
plt.ylabel('Probability')
plt.xticks([r + bar_width for r in range(len(keys))], keys)
#plt.legend(loc = "upper left", ncol=1)
plt.savefig(exp_folder + "images/Grover_shots.pdf")
plt.close()

# Heatmaps
import numpy as np
squared_error_rl = np.abs(np.square(dm_truth - dm_rl))
squared_error_rb = np.abs(np.square(dm_truth - dm_RB))

fig, axs = plt.subplots(1, 2, figsize=(16, 9))

# Plot the heatmaps
cax1 = axs[0].imshow(squared_error_rl, cmap='viridis')
fig.colorbar(cax1, ax=axs[0])
axs[0].set_title('MSE Truth-RL')

cax2 = axs[1].imshow(squared_error_rb, cmap='viridis')
fig.colorbar(cax2, ax=axs[1])
axs[1].set_title('MSE Truth-RB')
plt.savefig(exp_folder + "images/Grover_heatmap.pdf")