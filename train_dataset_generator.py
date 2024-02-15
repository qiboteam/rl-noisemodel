import os
import numpy as np
from rlnoise.dataset import Dataset, CircuitRepresentation
from rlnoise.custom_noise import CustomNoiseModel


#benchmark_circ_path=os.getcwd()+'/src/rlnoise/bench_dataset/'

benchmark_circ_path = 'enanched/'
config_file = "config.json"

if not os.path.exists(benchmark_circ_path):
    os.makedirs(benchmark_circ_path)

noise_model = CustomNoiseModel(config_file)
rep = CircuitRepresentation(config_file)


number_of_gates_per_qubit=[20]
qubits=3
number_of_circuits=500
dataset_name='test_set_enhanced'
enhanced_dataset = True

for i in range(9):
    dataset_name= f'test_set_enhanced{i}'
    with open(benchmark_circ_path+dataset_name+"_D%d_%dQ_len%d.npz"%(i,qubits,number_of_circuits),"wb") as f:
        nqubits = qubits
        depth = i
        ncirc = number_of_circuits
        dataset = Dataset(
            config_file,
            n_circuits = ncirc,
            n_gates = depth,
            n_qubits = nqubits,
            representation = rep,
            enhanced_dataset = enhanced_dataset,
            clifford = True,
            shadows = False,
            noise_model = noise_model,
            mode = 'rep',
            # backend = "IBM"
        )
        train_set=np.array(dataset.train_circuits)
        train_label=np.array(dataset.train_noisy_label)
        val_set=np.array(dataset.val_circuits)
        val_label=np.array(dataset.val_noisy_label)
        np.savez(f,train_set=train_set, 
                    train_label=train_label, 
                    val_label=val_label, 
                    val_set=val_set, allow_pickle=True)



