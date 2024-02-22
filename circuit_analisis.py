from qibo import gates, Circuit
import numpy as np
from rlnoise.utils import RL_NoiseModel
from stable_baselines3 import PPO
from rlnoise.custom_noise import CustomNoiseModel 
from rlnoise.dataset import CircuitRepresentation

dataset_raw = np.load("src/rlnoise/simulation_phase/3Q_non_clifford/non_clifford_set.npz", allow_pickle=True)
circuits = dataset_raw["train_circ"]
rep = CircuitRepresentation()

initial_rx = 0
initial_rz = 0
for circuit in circuits:
    circuit = rep.rep_to_circuit(circuit)
    for gate in circuit.queue:
        if isinstance(gate, gates.RX):
            initial_rx += 1
        elif isinstance(gate, gates.RZ):
            initial_rz += 1

damping = 0
dep = 0
rx = 0
rz = 0

agent = PPO.load("src/rlnoise/simulation_phase/3Q_random_Clifford/3Q_D7_Simulation5000.zip")
noise_model = CustomNoiseModel("src/rlnoise/config.json")
rl_noise = RL_NoiseModel(agent=agent, circuit_representation=rep)

for circuit in circuits:
    circuit = rep.rep_to_circuit(circuit)
    rl_circuit = rl_noise.apply(circuit)
    for gate in rl_circuit.queue:
        if isinstance(gate, gates.ResetChannel):
            damping += 1
        elif isinstance(gate, gates.DepolarizingChannel):
            dep += 1
        elif isinstance(gate, gates.RX):
            rx += 1
        elif isinstance(gate, gates.RZ):
            rz += 1

print("RL")
print("Damping: ", damping/400)
print("Depolarizing: ", dep/400)
print("RX: ", (rx-initial_rx)/400)
print("RZ: ", (rz-initial_rz)/400)

damping = 0
dep = 0
rx = 0
rz = 0

for circuit in circuits:
    circuit = rep.rep_to_circuit(circuit)
    gt_circuit = noise_model.apply(circuit)
    for gate in rl_circuit.queue:
        if isinstance(gate, gates.ResetChannel):
            damping += 1
        elif isinstance(gate, gates.DepolarizingChannel):
            dep += 1
        elif isinstance(gate, gates.RX):
            rx += 1
        elif isinstance(gate, gates.RZ):
            rz += 1

print("Ground Truth")
print("Damping: ", damping/400)
print("Depolarizing: ", dep/400)
print("RX: ", (rx-initial_rx)/400)
print("RZ: ", (rz-initial_rz)/400)



