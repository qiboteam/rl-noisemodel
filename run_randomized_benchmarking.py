import argparse, os, json
import numpy as np
from rlnoise.utils import randomized_benchmarking
from rlnoise.dataset import CircuitRepresentation
from rlnoise.custom_noise import CustomNoiseModel
from qibo.noise import NoiseModel, DepolarizingError


parser = argparse.ArgumentParser(description='Runs randomized benchmarking.')
parser.add_argument('--dataset')
parser.add_argument('--agent')

args = parser.parse_args()
assert args.dataset is not None, "Specify the path to the dataset dir."

rep = CircuitRepresentation()

circuits = []
for file in os.listdir(args.dataset):
    if file[-4:] == ".npz":
        with open(f"{args.dataset}/{file}", 'rb') as f:
            for c in np.load(f, allow_pickle=True)['clean_rep']:
                circuits.append(rep.rep_to_circuit(c))

if os.path.isfile(f"{args.dataset}/config.json"):
    with open(f"{args.dataset}/config.json", 'r') as f:
        conf = json.load(f)
    noise_conf = conf['noise']
    noise_model = CustomNoiseModel(
        primitive_gates = noise_conf['primitive_gates'],
        lam = noise_conf['dep_lambda'], 
        p0 = noise_conf['p0'],
        epsilon_x = noise_conf['epsilon_x'],
        epsilon_z = noise_conf['epsilon_z'],
        x_coherent_on_gate = noise_conf['x_coherent_on_gate'],
        z_coherent_on_gate = noise_conf['z_coherent_on_gate'],
        damping_on_gate = noise_conf['damping_on_gate'],
        depol_on_gate = noise_conf['depol_on_gate'],
    )
else:
    noise_model = None

depths, survival_probs, err, optimal_params, model = randomized_benchmarking(circuits, noise_model=noise_model)

import matplotlib.pyplot as plt
plt.errorbar(depths, survival_probs, yerr=err, fmt="o", elinewidth=1, capsize=3, c='orange')
plt.plot(depths, model(depths), c='orange')

import matplotlib.patches as mpatches
patches = [mpatches.Patch(color='orange', label=f"Decay: {optimal_params[1]:.2f}")]

# Build a Depolarizing toy model
depolarizing_toy_model = NoiseModel()
depolarizing_toy_model.add(DepolarizingError(1 - optimal_params[1]))
_, survival_probs, err, optimal_params, model = randomized_benchmarking(circuits, noise_model=depolarizing_toy_model)

plt.errorbar(depths, survival_probs, yerr=err, fmt="o", elinewidth=1, capsize=3, c='blue')
plt.plot(depths, model(depths+np.ones(len(depths))), c='blue')
patches.append(mpatches.Patch(color='blue', label=f"Depolarizing toy model, Decay: {optimal_params[1]:.2f}"))

plt.legend(handles=patches)

plt.ylabel('Survival Probability')
plt.xlabel('Depth')
plt.show()

