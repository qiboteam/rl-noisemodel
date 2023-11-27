import argparse, os, json
import numpy as np
from rlnoise.utils import randomized_benchmarking
from rlnoise.dataset import CircuitRepresentation
from rlnoise.custom_noise import CustomNoiseModel
from qibo.noise import NoiseModel, DepolarizingError


parser = argparse.ArgumentParser(description='Runs randomized benchmarking.')
parser.add_argument('--dataset')
parser.add_argument('--agent')
parser.add_argument('--backend', default=None)
parser.add_argument('--platform', default=None)

args = parser.parse_args()

if args.backend is not None:
    from qibo import set_backend

    set_backend(args.backend, platform=args.platform)

nqubits = 3
args.dataset = f'src/rlnoise/simulation_phase/RB/{nqubits}Q/dataset/'
#args.agent = f'src/rlnoise/simulation_phase/{nqubits}Q_training/3Q_D7_AllNoises_LogReward_798000.zip'

assert args.dataset is not None, "Specify the path to the dataset dir."

rep = CircuitRepresentation()

circuits = []
for file in os.listdir(args.dataset):
    if file[-4:] == ".npz":
        with open(f"{args.dataset}/{file}", 'rb') as f:
            for c in np.load(f, allow_pickle=True)['clean_rep']:
                circuits.append(rep.rep_to_circuit(c))
                


noise_model = None if args.backend == 'qibolab' else CustomNoiseModel()

depths, survival_probs, err, optimal_params, model = randomized_benchmarking(circuits, noise_model=noise_model)

import matplotlib.pyplot as plt
plt.errorbar(depths, survival_probs, yerr=err, fmt="o", elinewidth=1, capsize=3, c='orange')
plt.plot(depths, model(depths), c='orange')

import matplotlib.patches as mpatches
patches = [mpatches.Patch(color='orange', label=f"True Noise, Decay: {optimal_params[1]:.2f}")]
from qibo.backends import NumpyBackend

# Build a Depolarizing toy model
depolarizing_toy_model = NoiseModel()
depolarizing_toy_model.add(DepolarizingError(1 - optimal_params[1]))
_, survival_probs, err, optimal_params, model = randomized_benchmarking(circuits, noise_model=depolarizing_toy_model, backend=NumpyBackend())

plt.errorbar(depths, survival_probs, yerr=err, fmt="o", elinewidth=1, capsize=3, c='blue')
plt.plot(depths, model(depths), c='blue')
patches.append(mpatches.Patch(color='blue', label=f"Depolarizing toy model, Decay: {optimal_params[1]:.2f}"))

if args.agent is not None:
    from rlnoise.utils import RL_NoiseModel
    from stable_baselines3 import PPO
    
    # load trained agent
    agent = PPO.load(args.agent)
    agent_noise_model = RL_NoiseModel(agent, rep)
    _, survival_probs, err, optimal_params, model = randomized_benchmarking(circuits, noise_model=agent_noise_model, backend=NumpyBackend())
    plt.errorbar(depths, survival_probs, yerr=err, fmt="o", elinewidth=1, capsize=3, c='green')
    plt.plot(depths, model(depths), c='green')
    patches.append(mpatches.Patch(color='green', label=f"RL Agent, Decay: {optimal_params[1]:.2f}"))

plt.legend(handles=patches)

plt.ylabel('Survival Probability')
plt.xlabel('Depth')
plt.savefig('RB.pdf', format='pdf', dpi=300)
plt.show()

