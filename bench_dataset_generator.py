import os
import numpy as np
from rlnoise.dataset import Dataset, CircuitRepresentation
from rlnoise.CustomNoise import CustomNoiseModel


benchmark_circ_path='src/rlnoise/bench_dataset'
if not os.path.exists(benchmark_circ_path):
    os.makedirs(benchmark_circ_path)

noise_model = CustomNoiseModel()

rep = CircuitRepresentation(
    primitive_gates = noise_model.primitive_gates,
    noise_channels = noise_model.channels,
    shape = '3d',
    coherent_noise=True
)

depths=[7]

for i in depths:
    f = open(benchmark_circ_path+"/depth_"+str(i)+"_3Q_CoherentOnly_100.npz","wb")
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


