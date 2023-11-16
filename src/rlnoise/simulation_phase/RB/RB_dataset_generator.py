import os
import numpy as np
from rlnoise.dataset import Dataset, CircuitRepresentation
from rlnoise.custom_noise import CustomNoiseModel

benchmark_circ_path = 'src/rlnoise/simulation_phase/RB/1Q/dataset/'

if not os.path.exists(benchmark_circ_path):
    os.makedirs(benchmark_circ_path)

noise_model = CustomNoiseModel()
rep = CircuitRepresentation()


number_of_gates_per_qubit=np.arange(3,31,3)
qubits=1
number_of_circuits=50
dataset_name='RB_set'


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
        noise_model = noise_model,
        mode = 'rep'
    )
    bench_label = np.asarray(dataset.get_dm_labels())
    bench_clean_representation = np.asarray(dataset.circ_rep)
    np.savez(f,label = bench_label, 
             clean_rep = bench_clean_representation,
             allow_pickle=True)

f.close()


