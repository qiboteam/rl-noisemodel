import os
from pathlib import Path
import numpy as np
from rlnoise.dataset import Dataset, CircuitRepresentation
from rlnoise.custom_noise import CustomNoiseModel


#benchmark_circ_path=os.getcwd()+'/src/rlnoise/bench_dataset/'
qubits=1
number_of_circuits=200

# benchmark_circ_path = f'src/rlnoise/simulation_phase/3Q_random_Clifford(heavy_noise)/{number_of_circuits}_circ/'
benchmark_circ_path = "src/rlnoise/"
config_file = f"{Path(__file__).parent}/src/rlnoise/simulation_phase/1Q_training_new/config.json"

if not os.path.exists(benchmark_circ_path):
    os.makedirs(benchmark_circ_path)

noise_model = CustomNoiseModel(config_file)
rep = CircuitRepresentation(config_file)


number_of_gates_per_qubit=[7]

dataset_name='Rand_cliff'
enhanced_dataset = False

for i in number_of_gates_per_qubit:
    with open(benchmark_circ_path+dataset_name+"_D%d_%dQ_len%d.npz"%(i,qubits,number_of_circuits),"wb") as f:
        nqubits = qubits
        ncirc = number_of_circuits
        dataset = Dataset(
            config_file,
            n_circuits = ncirc,
            n_gates = i,
            n_qubits = nqubits,
            representation = rep,
            enhanced_dataset = enhanced_dataset,
            clifford = False,
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



