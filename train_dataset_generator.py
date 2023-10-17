import os
import numpy as np
from rlnoise.dataset import Dataset, CircuitRepresentation
from rlnoise.custom_noise import CustomNoiseModel


#benchmark_circ_path=os.getcwd()+'/src/rlnoise/bench_dataset/'

benchmark_circ_path=os.getcwd()+'/src/rlnoise/bench_dataset/'

if not os.path.exists(benchmark_circ_path):
    os.makedirs(benchmark_circ_path)

noise_model = CustomNoiseModel()
rep = CircuitRepresentation()


number_of_gates_per_qubit=[15]
qubits=3
number_of_circuits=500
dataset_name='IBM_train'


for i in number_of_gates_per_qubit:
    f = open(benchmark_circ_path+dataset_name+"_D%d_%dQ_len%d.npz"%(i,qubits,number_of_circuits),"wb")
    nqubits = qubits
    depth = i
    ncirc = number_of_circuits
    dataset = Dataset(
        n_circuits = ncirc,
        n_gates = depth,
        n_qubits = nqubits,
        representation = rep,
        clifford = True,
        shadows = False,
        noise_model = None,
        mode = 'rep',
        backend="IBM"
    )
    train_set=np.asarray(dataset.train_circuits)
    #train_label=np.asarray(dataset.train_noisy_label)
    val_set=np.asarray(dataset.val_circuits)
    #val_label=np.asarray(dataset.val_noisy_label)
    np.savez(f,train_set=train_set, val_set=val_set,allow_pickle=True)

f.close()


