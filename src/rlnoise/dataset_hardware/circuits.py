import os
import numpy as np
import copy
from qibo import set_backend
from rlnoise.dataset import CircuitRepresentation
from rlnoise.rewards.state_tomography import StateTomography

benchmark_circ_path=os.getcwd()+'/'
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

set_backend('qibolab','tii1q_b1')

rhos = []
for circ in qibo_training_circuits:

    st = StateTomography()

    st.get_circuits(circ)
    st.meas_obs(noise=None,readout_mit=False)
    rho = st.get_rho()

    st.meas_obs(noise=None,readout_mit=True)
    cal_mat = st.cal_mat
    rho_mit = st.get_rho()

    circ.density_matrix = True
    rho_exact = circ().state()

    rhos.append([circ, rho_exact, rho, rho_mit, cal_mat])


np.save('density_matrices.npy',rhos)