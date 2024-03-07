import numpy as np
from rlnoise.dataset import CircuitRepresentation

rep = CircuitRepresentation()
with open('src/rlnoise/hardware_test/dm_1Q_TII/density_matrices_training2.npy', 'rb') as f:
    tmp1 = np.load(f, allow_pickle=True)
    train_circ = np.array([rep.circuit_to_array(tmp1[i,0]) for i in range(tmp1.shape[0])])
    train_label = np.array([tmp1[i,3] for i in range(tmp1.shape[0])])

with open('src/rlnoise/hardware_test/dm_1Q_TII/density_matrices_validation2.npy', 'rb') as f1:
    tmp2 = np.load(f1, allow_pickle=True)
    val_circ = np.array([rep.circuit_to_array(tmp2[i,0]) for i in range(tmp2.shape[0])])
    val_label = np.array([tmp2[i,3] for i in range(tmp2.shape[0])])

with open('src/rlnoise/hardware_test/dm_1Q_TII/dataset_1Q_D5_TII.npz', 'wb') as f2:
    np.savez(f2, train_set=train_circ, 
             train_label=train_label, 
             val_set=val_circ, 
             val_label=val_label)