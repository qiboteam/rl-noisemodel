from qibo import Circuit, gates
import random
from pathlib import Path
from rlnoise.custom_noise import string_to_gate
from inspect import signature
import numpy as np


def generate_random_circuit(depth):
    """Generate a random circuit."""
    circuit = Circuit(1, density_matrix=True)
    for _ in range(depth):
        gate = string_to_gate(random.choice(["RX", "RZ"]))
        # 2 qubit gate
        theta = random.choice([0, 0.25, 0.5, 0.75]) if gate is gates.RZ else 0.25
        theta *= 2 * np.pi
        circuit.add(gate(0, theta=theta))

    return circuit

circ_list = []
for i in range(200):
    circ_list.append(generate_random_circuit(depth=15))

dataset_name = f"{Path(__file__).parent}/200_circ_set.npy"
np.save(dataset_name, arr=np.array(circ_list))

## Save RB set


# for i in range(3,33,3):
#     data_name = f"{Path(__file__).parent}/new_datasets/RB_set/D{i}_len50"
#     circ_list = []
#     for i in range(50): 
#         circ_list.append(generate_random_circuit(depth=i))
#     np.save(data_name, arr=np.array(circ_list))
