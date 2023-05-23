import os
import numpy as np
from rlnoise.dataset import Dataset, CircuitRepresentation
from rlnoise.custom_noise import CustomNoiseModel
from configparser import ConfigParser

benchmark_circ_path=os.getcwd()+'/src/rlnoise/bench_dataset/'
if not os.path.exists(benchmark_circ_path):
    os.makedirs(benchmark_circ_path)

params=ConfigParser()
params.read("src/rlnoise/config.ini")

noise_model = CustomNoiseModel(primitive_gates=params.get('noise','primitive_gates'),lam=params.get('noise','dep_lambda'),p0=params.get('noise','p0'),x_coherent_on_gate=['rx'],z_coherent_on_gate=['rz'],epsilon_x=params.get('noise','epsilon_x'),epsilon_z=params.get('noise','epsilon_z'),damping_on_gate=params.get('noise','damping_on_gate'),depol_on_gate=params.get('noise','depol_on_gate'))
rep = CircuitRepresentation()

number_of_gates_per_qubit=[5]
qubits=1
number_of_circuits=1000
dataset_name='Coherent-on_Std-on'

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
    train_set=np.asarray(dataset.train_circuits)
    train_label=np.asarray(dataset.train_noisy_label)
    val_set=np.asarray(dataset.val_circuits)
    val_label=np.asarray(dataset.val_noisy_label)
    np.savez(f,train_set=train_set, train_label=train_label, val_set=val_set,val_label=val_label,allow_pickle=True)

f.close()


