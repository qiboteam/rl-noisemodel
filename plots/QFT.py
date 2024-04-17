from rlnoise.noise_model import CustomNoiseModel
from rlnoise.randomized_benchmarking import fill_identity
from rlnoise.metrics import compute_fidelity
from rlnoise.rl_agent import Agent
from rlnoise.gym_env import QuantumCircuit
from rlnoise.utils import qft, unroll_circuit
from qibo.models import Circuit
from qibo import gates
from qibo.noise import NoiseModel, DepolarizingError

exp_folder = "simulation/experiments/3q_high_noise/"
model_file = exp_folder + "model.zip"
config_file = exp_folder + "config.json"
dataset_file = exp_folder + "dataset.npz"
lambda_rb = 0.12

circuit = qft()
circuit = unroll_circuit(circuit)
noise_model = CustomNoiseModel(config_file=config_file)
noisy_circuit = noise_model.apply(circuit)

env = QuantumCircuit(dataset_file = dataset_file, config_file = config_file)
rl_noise_model = Agent(config_file = config_file, env = env, model_file_path = model_file)
rl_noisy_circuit = rl_noise_model.apply(circuit)

noise = NoiseModel()
noise.add(DepolarizingError(lambda_rb))
rb_processed_circuit = fill_identity(circuit)
RB_noisy_circuit = noise.apply(rb_processed_circuit)

print("Length: ", len(circuit.queue))
print("Moments: ", len(circuit.queue.moments))
print("No noise", compute_fidelity(noisy_circuit().state(), circuit().state()))
print("RL agent", compute_fidelity(noisy_circuit().state(), rl_noisy_circuit().state()))
print("RB noise", compute_fidelity(noisy_circuit().state(), RB_noisy_circuit().state()))

def copy_circ(circ):
    new_circ = Circuit(3, density_matrix=True)
    for gate in circ.queue:
        new_circ.add(gate)
    return new_circ       

final_circuit2 = copy_circ(circuit)
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

# if test_only_depol_model is True:
#     rl_noise_only_dep = RL_NoiseModel(agent = agent_depol, circuit_representation =  CircuitRepresentation(config_depol_agent))
#     rl_dep_noisy_circuit = rl_noise_only_dep.apply(final_circuit)
#     print("RL depol agent", compute_fidelity(noisy_circuit().state(), rl_dep_noisy_circuit().state()))
#     rl_dep_noisy_circuit2 = copy_circ(rl_dep_noisy_circuit)
#     rl_dep_noisy_circuit2.add(gates.M(0,1,2))
#     rl_dep_shots = rl_dep_noisy_circuit2.execute(nshots=10000)
#     rl_dep_shots = dict(sorted(dict(rl_dep_shots.frequencies()).items()))
#     values4 = list(rl_dep_shots.values())
#     r4 = [x + bar_width for x in r3]

fig=plt.figure(figsize=(12, 9))
ax=fig.add_subplot(111)
# Create the bar plot
ax.bar(r1, values1, width=bar_width, label='Ground truth noise', color='#e60049')
ax.bar(r2, values2, width=bar_width, label='RL (standard)', color='#0bb4ff')
ax.bar(r3, values3, width=bar_width, label='RB', color='green')
# if test_only_depol_model:
#     ax.bar(r4, values4, width=bar_width, label='RL (only dep)', color='orange')

# Customize the plot
plt.xlabel('Result')
plt.ylabel('Counts')
plt.ylim(0, 3000)
plt.xticks([r + bar_width for r in range(len(keys))], keys)
plt.legend(loc = "upper left", ncol=1)
plt.savefig("QFT_shots.pdf", )
plt.show()