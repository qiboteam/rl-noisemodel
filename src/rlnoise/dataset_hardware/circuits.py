import os
import numpy as np
import copy
from rlnoise.dataset import CircuitRepresentation
benchmark_circ_path=os.getcwd()+'/src/rlnoise/dataset_hardware/'
bench_name='hardware_len500_D5_1Q_len500.npz'

f = open(benchmark_circ_path+bench_name,"rb")
tmp=np.load(f,allow_pickle=True)

train_set=copy.deepcopy(tmp['train_set'])
val_set=copy.deepcopy(tmp['val_set'])
print(val_set.shape)
print(train_set.shape)

#Those lists contain 400 and 100 qibo circuits respectively 
qibo_training_circuits=[CircuitRepresentation().rep_to_circuit(train_set[i]) for i in range(train_set.shape[0])]
qibo_validation_circuits=[CircuitRepresentation().rep_to_circuit(val_set[i]) for i in range(val_set.shape[0])]
print(qibo_training_circuits[0].draw())
print(qibo_validation_circuits[0].draw())