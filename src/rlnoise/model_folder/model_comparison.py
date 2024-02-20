import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from rlnoise.utils import model_evaluation, RL_NoiseModel
from rlnoise.custom_noise import CustomNoiseModel
from rlnoise.metrics import compute_fidelity
from rlnoise.dataset import CircuitRepresentation
from qibo import gates, Circuit
from qibo.noise import NoiseModel
# non_cliff_set = np.load(f"{Path(__file__).parents[1]}/simulation_phase/3Q_non_clifford/non_clifford_set.npz", allow_pickle=True)
# test_circ = non_cliff_set["train_circ"]
# test_labels = non_cliff_set["train_label"]


model_path = "src/rlnoise/model_folder/Rans_cliff_mse_tanh/3Q_Rand_clif_mse_tanh12000.zip"
agent = PPO.load(model_path)



# evaluation_results = model_evaluation(test_circ, test_labels, agent)

# print(f"Fid:{evaluation_results['fidelity']}, Fid_std:{evaluation_results['fidelity_std']}, TD:{evaluation_results['trace_distance']}, TD_std:{evaluation_results['trace_distance_std']}")

# Old_random_generator: Fid:[0.969], Fid_std:[0.014], TD:[0.148], TD_std:[0.032]
# Rand_cliff_generator: Fid:[0.973], Fid_std:[0.017], TD:[0.136], TD_std:[0.04]

# Old_random_generator + Tanh: Fid:[0.972], Fid_std:[0.012], TD:[0.14], TD_std:[0.032]
# Rand_cliff_generator + Tanh: Fid:[0.973], Fid_std:[0.012], TD:[0.137], TD_std:[0.031]

def grover():
    """Creates a Grover circuit with 3 qubits.
    The circuit searches for the 11 state, the last qubit is ancillary"""
    circuit = Circuit(3, density_matrix=True)
    circuit.add(gates.RZ(0, np.pi/2))
    circuit.add(gates.RX(0, np.pi/2))
    circuit.add(gates.RZ(0, np.pi/2))
    circuit.add(gates.RZ(1, np.pi/2))
    circuit.add(gates.RX(1, np.pi/2))
    circuit.add(gates.RZ(1, np.pi/2))
    circuit.add(gates.RX(2, np.pi))
    circuit.add(gates.RZ(2, np.pi/2))
    circuit.add(gates.RX(2, np.pi/2))
    circuit.add(gates.RZ(2, np.pi/2))
    #Toffoli
    circuit.add(gates.CZ(1, 2))
    circuit.add(gates.RX(2, -np.pi / 4))
    circuit.add(gates.CZ(0, 2))
    circuit.add(gates.RX(2, np.pi / 4))
    circuit.add(gates.CZ(1, 2))
    circuit.add(gates.RX(2, -np.pi / 4))
    circuit.add(gates.CZ(0, 2))
    circuit.add(gates.RX(2, np.pi / 4))
    circuit.add(gates.RZ(1, np.pi / 4))
    circuit.add(gates.RZ(1, np.pi/2))
    circuit.add(gates.RX(1, np.pi/2))
    circuit.add(gates.RZ(1, np.pi/2))
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.RZ(0, np.pi / 4))
    circuit.add(gates.RX(1, -np.pi / 4))
    circuit.add(gates.CZ(0, 1))
    ###
    circuit.add(gates.RZ(0, np.pi/2))
    circuit.add(gates.RX(0, np.pi/2))
    circuit.add(gates.RZ(0, np.pi/2))
    circuit.add(gates.RX(0, np.pi))
    circuit.add(gates.RX(1, np.pi))
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.RX(0, np.pi))
    circuit.add(gates.RX(1, np.pi))
    circuit.add(gates.RZ(0, np.pi/2))
    circuit.add(gates.RX(0, np.pi/2))
    circuit.add(gates.RZ(0, np.pi/2))
    circuit.add(gates.RZ(1, np.pi/2))
    circuit.add(gates.RX(1, np.pi/2))
    circuit.add(gates.RZ(1, np.pi/2))
    return circuit

def copy_circ(circ):
    new_circ = Circuit(3, density_matrix=True)
    for gate in circ.queue:
        new_circ.add(gate)
    return new_circ

rep = CircuitRepresentation()

grover_circ = grover()
grover_circ2 = grover()
grover_circ2.add(gates.M(0,1,2))
grover_label = grover_circ().state()

grover_rep = rep.circuit_to_array(grover_circ)

results = grover_circ2.execute(nshots=1000)
print(results.frequencies())

noise_model = CustomNoiseModel(config_file=f"{Path(__file__).parents[1]}/config.json")
noisy_circuit = noise_model.apply(grover_circ)
noisy_circuit.add(gates.M(0,1,2))

rl_noise = RL_NoiseModel(agent, rep)

noisy_grover2 = rl_noise.apply(grover_circ)
noisy_grover2.add(gates.M(0,1,2))
print(noisy_grover2.execute(nshots=1000).frequencies())
print(noisy_circuit.execute(nshots=1000).frequencies())

# noisy_grover = rl_noise.apply(grover_circ)
# print(compute_fidelity(noisy_grover().state(), noisy_grover2().state()))

# grover_labels = grover_circ2().state()
# print(grover_labels.shape)
# eval_grover = model_evaluation(grover_rep.reshape((1,25,3,8)), grover_labels.reshape((1,8,8)), agent)
# print(eval_grover['fidelity'])