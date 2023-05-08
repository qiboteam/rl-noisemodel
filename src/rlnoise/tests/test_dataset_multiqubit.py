
import time
from rlnoise.datasetv2 import Dataset, CircuitRepresentation
import numpy as np
from rlnoise.CustomNoise import CustomNoiseModel

nqubits = 2
depth = 5
ncirc = 1

noise_model = CustomNoiseModel()

rep = CircuitRepresentation(
    primitive_gates = noise_model.primitive_gates,
    noise_channels = noise_model.channels,
    shape = '3d',
    coherent_noise=True
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
                           #testing rep_to_circuit,set_mode and getitem() of dataset

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
        [1, 0, 0, 0.75, 0.1, 0], 
        [0, 1, 0, 0.5, 0, 0.05]
        ],
        [
        [0, 0, 1, 0, 0.1, 0],
        [0, 0, 0, 0, 0.1, 0], 
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
        [0, 1, 0, 0.75, 0, 0,0,0.1], 
        [0, 1, 0, 0.5, 0, 0,0,0.1]
        ],
        [
        [0, 0, 1, 0, 0.1, 0,1,1], 
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
    
    print('------Clean test circ %d ------'%(i))
    print(test_circ.draw())
    print('------Noisy test circ %d------ '%(i))
    print(noisy_test_circ.draw())
    #print('------test rep %d ------\n'%(i),test_rep)
    print('------noisy test rep ------ \n',noisy_test_rep2)   
    print('Reconstructed Noisy circuit:')
    print(reconstructed_test.draw(),'\n')

    #print(reconstructed_noisy_circuit.draw())
    
    #print('\n Difference between real dm_label and reconstructed: \n',np.float32((np.square(noisy_test_circ().state()-reconstructed_noisy_circuit().state())).mean())) 
#print('reconstructed_noisy_circuit:\n')
#print(reconstructed_test.draw())



                    #testing train and validatio split
'''                    
train_set=np.asarray(dataset.train_circuits)
train_label=np.asarray(dataset.train_noisy_label)
val_set=np.asarray(dataset.val_circuits)
val_label=np.asarray(dataset.val_noisy_label)
dm_difference=0.
for idx,train_rep in enumerate(train_set):
    train_circ=noise_model.apply(rep.rep_to_circuit(train_rep))
    dm_difference+=np.mean(train_circ().state()-train_label[idx])
for idx,val_rep in enumerate(val_set):
    val_circ=noise_model.apply(rep.rep_to_circuit(val_rep))
    dm_difference+=np.mean(val_circ().state()-val_label[idx])
if dm_difference ==0:
    print('----Test dataset train_val_split: PASSED----')


print('Execution time: %f seconds'%(end_time-start_time))
'''


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