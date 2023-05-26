import sys
from rlnoise.dataset import Dataset, CircuitRepresentation
import numpy as np
from qibo.noise import DepolarizingError, NoiseModel
from qibo import gates
from rlnoise.custom_noise import CustomNoiseModel

nqubits = 2
depth = 5
ncirc = 2
val_split = 0.2

noise_model = CustomNoiseModel()


rep = CircuitRepresentation()

# create dataset
dataset = Dataset(
    n_circuits = ncirc,
    n_gates = depth,
    n_qubits = nqubits,
    representation = rep,
    clifford = True,
    shadows = False,
    readout_mit = False,
    noise_model = noise_model,
    mode = 'rep',
    backend= None,
)

# input circuit
circuit_rep = dataset[0]
dataset.set_mode('circ')
circuit = dataset[0]
dataset.set_mode('noisy_circ')
noisy_circuit = dataset[0]

dm=dataset.get_dm_labels()
print("DM: ", dm)

def test_representation():
    print('> Noiseless Circuit:\n')
    print(circuit.draw())
    array = rep.circuit_to_array(circuit)
    print(' --> Representation:\n', array)
    print(' --> Circuit Rebuilt:\n')
    print(rep.rep_to_circuit(array).draw())
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('> Noisy Circuit:\n')
    print(noisy_circuit.draw())

test_representation()