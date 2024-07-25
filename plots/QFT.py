from rlnoise.noise_model import CustomNoiseModel
from rlnoise.randomized_benchmarking import fill_identity
from rlnoise.rl_agent import Agent
from rlnoise.gym_env import QuantumCircuit
from rlnoise.utils import qft, mms, mse, compute_fidelity
from qibo.models import Circuit
import numpy as np
from qibo.noise import NoiseModel, DepolarizingError
import json

exp_folder = "experiments/simulation/3q_high/"
model_file = exp_folder + "model.zip"
config_file = exp_folder + "config.json"
dataset_file = exp_folder + "dataset.npz"
# High noise: 0.0332
# Low noise: 0.0156
lambda_rb = 0.0332

circuit = qft()
print(circuit.draw())
noise_model = CustomNoiseModel(config_file=config_file)
noisy_circuit = noise_model.apply(circuit)

env = QuantumCircuit(dataset_file = dataset_file, config_file = config_file)
rl_noise_model = Agent(config_file = config_file, env = env, model_file_path = model_file)
rl_noisy_circuit = rl_noise_model.apply(circuit)
print(rl_noisy_circuit.draw())

noise = NoiseModel()
noise.add(DepolarizingError(lambda_rb))
rb_processed_circuit = fill_identity(circuit)
RB_noisy_circuit = noise.apply(rb_processed_circuit)

dm_truth = noisy_circuit().state()
dm_rl = rl_noisy_circuit().state()
dm_RB = RB_noisy_circuit().state()

print("Circuit Info: ")
print("Length: ", len(circuit.queue))
print("Moments: ", len(circuit.queue.moments))
print("Fidelity: ")
print("No noise: ", compute_fidelity(dm_truth, circuit().state()))
print("RL agent: ", compute_fidelity(dm_truth, dm_rl))
print("RB noise: ", compute_fidelity(dm_truth, dm_RB))
print("MMS: ", compute_fidelity(dm_truth, mms(8)))
print("MSE: ")
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
with open(exp_folder + "images/QFT_result.json", "w") as f:
    json.dump(result_dict, f)

def copy_circ(circ):
    new_circ = Circuit(3, density_matrix=True)
    for gate in circ.queue:
        new_circ.add(gate)
    return new_circ       

def compute_probabilities(rho):
    probs = {}
    for i in range(8):
        probs[format(i, "03b")] = np.abs(rho[i, i])
    return probs
            
no_noise_shots = compute_probabilities(circuit().state())
noise_shots = compute_probabilities(dm_truth)
rl_shots = compute_probabilities(dm_rl)
RB_shots = compute_probabilities(dm_RB)

print("Shots:")
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
# Create the bar plot
ax.bar(r1, values1, width=bar_width, label='Ground truth', color='#e60049')
ax.bar(r2, values2, width=bar_width, label='RL', color='#0bb4ff')
ax.bar(r3, values3, width=bar_width, label='RB', color='green')

plt.xlabel('State')
plt.ylabel('Probability')
plt.xticks([r + bar_width for r in range(len(keys))], keys)
plt.legend(loc = "upper right", ncol=3)
plt.ylim(0, 0.2)
plt.savefig(exp_folder + "images/QFT_shots.pdf" )
plt.close()

# Heatmaps
import numpy as np
squared_error_rl = np.abs(dm_truth - dm_rl)
squared_error_rb = np.abs(dm_truth - dm_RB)

# Determine global min and max values for consistent color scaling
vmin = min(squared_error_rl.min(), squared_error_rb.min())
vmax = max(squared_error_rl.max(), squared_error_rb.max())

fig, axs = plt.subplots(1, 2, figsize=(22, 9))

color = 'plasma'

cax1 = axs[0].imshow(squared_error_rl, cmap=color, vmin=vmin, vmax=vmax)
axs[0].set_title('RL')
axs[0].set_xticks([])
axs[0].set_yticks([])

cax2 = axs[1].imshow(squared_error_rb, cmap=color, vmin=vmin, vmax=vmax)
axs[1].set_title('RB')
axs[1].set_xticks([])
axs[1].set_yticks([])

# Create a colorbar at the bottom of the plots
cbar = fig.colorbar(cax1, ax=axs, orientation='vertical', fraction=0.09, pad=0.04)

plt.savefig(exp_folder + "images/QFT_heatmap.pdf")