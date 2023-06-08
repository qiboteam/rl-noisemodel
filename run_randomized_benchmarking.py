import argparse, os, json
import numpy as np
from rlnoise.utils import randomized_benchmarking
from rlnoise.dataset import CircuitRepresentation
from rlnoise.custom_noise import CustomNoiseModel

parser = argparse.ArgumentParser(description='Runs randomized benchmarking.')
parser.add_argument('--dataset')

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
    #noise_model = CustomNoiseModel()
else:
    noise_model = None

depths, survival_probs, err, optimal_params, model = randomized_benchmarking(circuits, noise_model=noise_model)

import matplotlib.pyplot as plt
plt.errorbar(depths, survival_probs, yerr=err, fmt="o", elinewidth=1, capsize=3)
plt.plot(depths, model(depths), c='orange')
plt.ylabel('Survival Probability')
plt.xlabel('Depth')

import matplotlib.patches as mpatches
patch = mpatches.Patch(color='orange', label=f"Decay: {optimal_params[1]:.2f}")
plt.legend(handles=[patch])
plt.show()
