import numpy as np
from pathlib import Path
import argparse
from rlnoise.dataset import CircuitRepresentation
from rlnoise.custom_noise import CustomNoiseModel
from rlnoise.metrics import compute_fidelity
from rlnoise.utils import RL_NoiseModel, unroll_circuit, grover, qft
from stable_baselines3 import PPO
from qibo.models import QFT, Circuit
from qibo import gates
from qibo.noise import NoiseModel, DepolarizingError

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=f'{Path(__file__).parent}/src/rlnoise/simulation_phase/3Q_random_Clifford(soft_noise_grover)/config_all_noise.json')
parser.add_argument('--model', type=str, default=f'{Path(__file__).parent}/src/rlnoise/simulation_phase/3Q_random_Clifford(soft_noise_grover)/3Q_Rand_clif_logmse175000_allNoise.zip')
args = parser.parse_args()

circuit_type = "Grover"
test_only_depol_model = True

agent = PPO.load(args.model)
agent_depol = PPO.load("src/rlnoise/simulation_phase/3Q_random_Clifford(soft_noise_grover)/3Q_Rand_clif_logmse_onlydepol340000.zip")
config_depol_agent = "src/rlnoise/simulation_phase/3Q_random_Clifford(soft_noise_grover)/config_only_dep.json"


if circuit_type == "QFT":
    circuit = qft()
    print(circuit.draw())
    final_circuit = unroll_circuit(circuit)
if circuit_type == "Grover":
    circuit = grover()
    final_circuit = circuit


noise_model = CustomNoiseModel(config_file=args.config)
noisy_circuit = noise_model.apply(final_circuit)

rl_noise_model = RL_NoiseModel(agent = agent, circuit_representation =  CircuitRepresentation(args.config))
rl_noisy_circuit = rl_noise_model.apply(final_circuit)

noise = NoiseModel()
noise.add(DepolarizingError(0.08))
RB_noisy_circuit = noise.apply(final_circuit)

print("Circuit type: ", circuit_type)
print("Length: ", len(final_circuit.queue))
print("Moments: ", len(final_circuit.queue.moments))
print("No noise", compute_fidelity(noisy_circuit().state(), final_circuit().state()))
print("RL agent", compute_fidelity(noisy_circuit().state(), rl_noisy_circuit().state()))
print("RB noise", compute_fidelity(noisy_circuit().state(), RB_noisy_circuit().state()))

def copy_circ(circ):
    new_circ = Circuit(3, density_matrix=True)
    for gate in circ.queue:
        new_circ.add(gate)
    return new_circ
        

final_circuit2 = copy_circ(final_circuit)
final_circuit2.add(gates.M(0,1,2))
noisy_circuit2 = copy_circ(noisy_circuit)
noisy_circuit2.add(gates.M(0,1,2))
rl_noisy_circuit2 = copy_circ(rl_noisy_circuit)
rl_noisy_circuit2.add(gates.M(0,1,2))
RB_noisy_circuit2 = copy_circ(RB_noisy_circuit)
RB_noisy_circuit2.add(gates.M(0,1,2))

no_noise_shots = final_circuit2.execute(nshots=10000)
noise_shots = noisy_circuit2.execute(nshots=10000)
rl_shots = rl_noisy_circuit2.execute(nshots=10000)
RB_shots = RB_noisy_circuit2.execute(nshots=10000)

no_noise_shots = dict(sorted(dict(no_noise_shots.frequencies()).items()))
noise_shots = dict(sorted(dict(noise_shots.frequencies()).items()))
rl_shots = dict(sorted(dict(rl_shots.frequencies()).items()))
RB_shots = dict(sorted(dict(RB_shots.frequencies()).items()))
if circuit_type == "Grover":
    for i in ("000", "001", "010", "011", "100", "101"):
        no_noise_shots[i] = 0
    no_noise_shots = dict(sorted(no_noise_shots.items()))


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

if test_only_depol_model is True:
    rl_noise_only_dep = RL_NoiseModel(agent = agent_depol, circuit_representation =  CircuitRepresentation(config_depol_agent))
    rl_dep_noisy_circuit = rl_noise_only_dep.apply(final_circuit)
    print("RL depol agent", compute_fidelity(noisy_circuit().state(), rl_dep_noisy_circuit().state()))
    rl_dep_noisy_circuit2 = copy_circ(rl_dep_noisy_circuit)
    rl_dep_noisy_circuit2.add(gates.M(0,1,2))
    rl_dep_shots = rl_dep_noisy_circuit2.execute(nshots=10000)
    rl_dep_shots = dict(sorted(dict(rl_dep_shots.frequencies()).items()))
    values4 = list(rl_dep_shots.values())
    r4 = [x + bar_width for x in r3]




fig=plt.figure(figsize=(16, 9))
ax=fig.add_subplot(111)
# Create the bar plot
ax.bar(r1, values1, width=bar_width, label='Ground truth noise', color='#e60049')
ax.bar(r2, values2, width=bar_width, label='RL (standard)', color='#0bb4ff')
ax.bar(r3, values3, width=bar_width, label='RB', color='green')
if test_only_depol_model:
    ax.bar(r4, values4, width=bar_width, label='RL (only dep)', color='orange')


# Customize the plot
plt.xlabel('Result')
plt.ylabel('Counts')
if circuit_type == "Grover":
    plt.ylim(0, 3500)
plt.xticks([r + bar_width for r in range(len(keys))], keys)
plt.legend(loc = "upper left", ncol=2)
plt.savefig(f"{circuit_type}_shots.pdf", )
plt.show()