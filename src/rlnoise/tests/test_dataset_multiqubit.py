import sys
from rlnoise.dataset import Dataset, CircuitRepresentation
import numpy as np
from qibo.noise import DepolarizingError, NoiseModel
from qibo import gates

nqubits = 3
depth = 5
ncirc = 10

noise_model = NoiseModel()
lam = 0.05
lamCZ=0.1
noise_model.add(DepolarizingError(lam), gates.RZ)
noise_model.add(DepolarizingError(lamCZ), gates.CZ)
noise_channel = gates.DepolarizingChannel((0,), lam=lam)
primitive_gates = ['RZ', 'RX','CZ']
channels = ['DepolarizingChannel']

rep = CircuitRepresentation(
    primitive_gates = primitive_gates,
    noise_channels = channels,
    shape = '3d'
)

# create dataset
dataset = Dataset(
    n_circuits = ncirc,
    n_gates = depth,
    n_qubits = nqubits,
    representation = rep,
    clifford = True,
    shadows = True,
    noise_model = noise_model,
    mode = 'rep'
)
for i in range(len(dataset[:])):
    dataset.set_mode('circ')
    test_circ=dataset[i]
    dataset.set_mode('noisy_circ')
    noisy_test_circ=dataset[i]
    dataset.set_mode('rep')
    test_rep=dataset[i]
    dataset.set_mode('noisy_rep')
    noisy_test_rep=dataset[i]
    reconstructed_circuit=rep.rep_to_circuit(test_rep)
    reconstructed_noisy_circuit=rep.rep_to_circuit(noisy_test_rep)
    '''
    print('------test circ %d ------\n'%(i))
    print(test_circ.draw())
    print('------noisy test circ %d------ \n\n'%(i))
    print(noisy_test_circ.draw())
    print('------test rep %d ------\n'%(i),test_rep)
    print('------noisy test rep %d------ \n'%(i),noisy_test_rep)   
    print('reconstructed_circuit:\n')
    print(reconstructed_circuit.draw())
    print('reconstructed_circuit:\n')
    print(reconstructed_noisy_circuit.draw())
    '''
    print('Difference between real dm_label and reconstructed: \n',(np.square(noisy_test_circ().state()-reconstructed_noisy_circuit().state())).mean()) #THERE IS A DIFFERENCE BETWEEN THE DM
"""
# input circuit
circuit_rep = dataset[0]
dataset.set_mode('circ')
circuit = dataset[0]
dataset.set_mode('noisy_circ')
noisy_circuit = dataset[0]

dm=dataset.get_dm_labels()
print("DM: ", dm)

def test_representation():
    print('> Noiseless Circuit:\n', circuit.draw())
    array = rep.circuit_to_array(circuit)
    print(' --> Representation:\n', array)
    print(' --> Circuit Rebuilt:\n', rep.array_to_circuit(array).draw())
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('> Noisy Circuit:\n', noisy_circuit.draw())
    array = rep.circuit_to_array(noisy_circuit)
    print(array)
    print(' --> Circuit Rebuilt:\n', rep.array_to_circuit(array).draw())

test_representation()
"""