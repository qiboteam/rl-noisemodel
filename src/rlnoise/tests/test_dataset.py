import sys
from rlnoise.dataset import Dataset, CircuitRepresentation
import numpy as np
from qibo.noise import DepolarizingError, NoiseModel
from qibo import gates
from rlnoise.custom_noise import CustomNoiseModel

nqubits = 1
depth = 5
ncirc = 10
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
    backend= None,
)

for i in range(ncirc):
    circ = dataset[i]
    print(len(circ))