
import time
from rlnoise.datasetv2 import Dataset, CircuitRepresentation
import numpy as np
from qibo.noise import DepolarizingError, NoiseModel, ThermalRelaxationError
from qibo import gates
from rlnoise.CustomNoise import CustomNoiseModel

nqubits = 3
depth = 3
ncirc = 10

noise_model = CustomNoiseModel()

rep = CircuitRepresentation(
    primitive_gates = noise_model.primitive_gates,
    noise_channels = noise_model.channels,
    shape = '3d'
)
start_time=time.time()
# create dataset
dataset = Dataset(
    n_circuits = ncirc,
    n_gates = depth,
    n_qubits = nqubits,
    representation = rep,
    clifford = True,
    shadows = False,
    noise_model = noise_model,
    mode = 'rep'
)
end_time = time.time()

for i in range(len(dataset[:])):
    dataset.set_mode('circ')
    test_circ=dataset[i]
    dataset.set_mode('noisy_circ')
    noisy_test_circ=dataset[i]
    dataset.set_mode('rep')
    test_rep=dataset[i]
    #dataset.set_mode('noisy_rep')
    #noisy_test_rep=dataset[i]
    reconstructed_circuit=rep.rep_to_circuit(test_rep)
    noisy_test_rep=np.asarray(
        [
        [
        [1, 0, 0, 0.5, 0.1, 0.05],
        [0, 0, 0, 0, 0, 0], 
        [0, 1, 0, 0.5, 0, 0.05]
        ],
        [
        [0, 0, 1, 0, 0.1, 0],
        [0, 0, 0, 0, 0, 0], 
        [0, 0, 1, 0, 0.1, 0.05]
        ],
        [
        [0, 0, 0, 0, 0.1, 0],
        [0, 1, 0, 0.5, 0.1, 0.05], 
        [0, 0, 0, 0, 0, 0.05]
        ],
        ]
    )
    noisy_test_rep2=np.asarray(
        [
        [
        [1, 0, 0, 0.5, 0.1, 0.05,0,0],
        [0, 0, 0, 0, 0, 0,0,0], 
        [0, 1, 0, 0.5, 0, 0.05,0,0]
        ],
        [
        [0, 0, 1, 0, 0.1, 0,0,0], #IT DOESENT ADD THE EPSILON GATES ON THE FIRST QUBIT OF CZ, SEE LINE 366 OF DATASETV2
        [0, 0, 0, 0, 0, 0,0,0], 
        [0, 0, 1, 0, 0.1, 0.05,0,0]
        ],
        [
        [0, 0, 0, 0, 0.1, 0,0,0],
        [0, 1, 0, 0.5, 0.1, 0.05,0,0], 
        [0, 0, 0, 0, 0, 0.05,1,1]
        ],
        ]
    )
    reconstructed_test=rep.rep_to_circuit(noisy_test_rep2)
    
    #reconstructed_noisy_circuit=rep.rep_to_circuit(noisy_test_rep)
    
    #print('------test circ %d ------\n'%(i))
    #print(test_circ.draw())
    print('------noisy test circ %d------ \n\n'%(i))
    print(noisy_test_circ.draw())
    print('------test rep %d ------\n'%(i),test_rep)
    #print('------noisy test rep %d------ \n'%(i),noisy_test_rep)   
    #print('reconstructed_circuit:\n')
    #print(reconstructed_circuit.draw())
    print('reconstructed_noisy_circuit:\n')
    print(reconstructed_test.draw())
    #print(reconstructed_noisy_circuit.draw())
    
    #print('\n Difference between real dm_label and reconstructed: \n',np.float32((np.square(noisy_test_circ().state()-reconstructed_noisy_circuit().state())).mean())) 
print('Execution time: %f seconds'%(end_time-start_time))

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