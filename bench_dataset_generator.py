import os
import numpy as np
from rlnoise.dataset import Dataset, CircuitRepresentation
from qibo.noise import DepolarizingError, NoiseModel, ThermalRelaxationError
from qibo import gates

noise_model = NoiseModel()
time = 0.07
lamCZ=0.15
noise_model.add(ThermalRelaxationError(t1=1,t2=1,time=time), gates.RZ)
noise_model.add(DepolarizingError(lamCZ), gates.CZ)

primitive_gates = ['RZ', 'RX','CZ']
channels = ['DepolarizingChannel','ThermalRelaxationChannel']

benchmark_circ_path=os.getcwd()+'/src/rlnoise/bench_dataset'
model_path=os.getcwd()+'/src/rlnoise/saved_models'
if not os.path.exists(benchmark_circ_path):
    os.makedirs(benchmark_circ_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

rep = CircuitRepresentation(
    primitive_gates = primitive_gates,
    noise_channels = channels,
    shape = '3d'
)

depths=[7]

for i in depths:
    f = open(benchmark_circ_path+"/depth_"+str(i)+".npz","wb")
    nqubits = 3
    depth = i
    ncirc = 100
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
    np.savez(f,train_set=train_set, train_label=train_label, val_set=val_set,val_label=val_label)

f.close()


