import sys
from rlnoise.dataset import Dataset, CircuitRepresentation
import numpy as np
from qibo.noise import DepolarizingError, NoiseModel
from qibo import gates
from rlnoise.custom_noise import CustomNoiseModel

nqubits = 3
depth = 10
ncirc = 100
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
    qibo_circ = rep.rep_to_circuit(circ)
    print("First circuit: ")
    print(qibo_circ.draw())
    circ2 = rep.circuit_to_array(qibo_circ)
    qibo_circ2 = rep.rep_to_circuit(circ2)
    print("second circuit: ")
    print(qibo_circ2.draw())
    #print(len(circ))
    #print('test n moments:',len(rep.rep_to_circuit(circ).queue.moments))
    #print(circ)