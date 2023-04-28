import os
import numpy as np
from rlnoise.datasetv2 import Dataset, CircuitRepresentation
from rlnoise.CustomNoise import CustomNoiseModel

noise_model = CustomNoiseModel()

benchmark_circ_path=os.getcwd()+'/src/rlnoise/bench_dataset'
model_path=os.getcwd()+'/src/rlnoise/saved_models'
if not os.path.exists(benchmark_circ_path):
    os.makedirs(benchmark_circ_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

rep = CircuitRepresentation(
    primitive_gates = noise_model.primitive_gates,
    noise_channels = noise_model.channels,
    shape = '3d',
    coherent_noise=False
)

depths=[5]

for i in depths:
    f = open(benchmark_circ_path+"/depth_"+str(i)+".npz","wb")
    nqubits = 3
    depth = i
    ncirc = 200
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


