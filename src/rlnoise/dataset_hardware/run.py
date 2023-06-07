import copy
import os

import joblib
import numpy as np
from qibo import gates
from qibo.backends import construct_backend

from rlnoise.dataset import CircuitRepresentation
from rlnoise.dataset_hardware.utils import classical_shadows, qiskit_state_tomography, state_tomography

benchmark_circ_path=os.getcwd()
bench_name='/hardware_len500_D5_1Q_len500.npz'

f = open(benchmark_circ_path+bench_name,"rb")
tmp=np.load(f,allow_pickle=True)

train_set=copy.deepcopy(tmp['train_set'])
val_set=copy.deepcopy(tmp['val_set'])
print(val_set.shape)
print(train_set.shape)

#Those lists contain 400 and 100 qibo circuits respectively 
qibo_training_circuits=[CircuitRepresentation().rep_to_circuit(train_set[i]) for i in range(train_set.shape[0])]
qibo_validation_circuits=[CircuitRepresentation().rep_to_circuit(val_set[i]) for i in range(val_set.shape[0])]
qibo_training_circuits[0]
qibo_validation_circuits[0]

#IBMProvider.save_account(token='')

qiskit = True
nshots = 100
shadow_size = 100
method = 'CS'
njobs = 2
backend_qibo = 'tii1q_b1'
backend_qiskit = 'ibmq_qasm_simulator'

if qiskit or method=='ST_qiskit':
    from qiskit_ibm_provider import IBMProvider
    provider = IBMProvider()
    backend = construct_backend('numpy')
    backend_qiskit = provider.get_backend(backend_qiskit)
else:
    from rlnoise.dataset_hardware.utils import rx_rule, x_rule
    os.environ["QIBOLAB_PLATFORMS"] = "/nfs/users/alejandro.sopena/qibolab_platforms_qrc/" #PATH to runcards
    backend = construct_backend('qibolab',backend_qibo)
    backend.compiler.__setitem__(gates.RX, rx_rule)
    backend.compiler.__setitem__(gates.X, x_rule)
    backend.transpiler = None
    backend_qiskit = None

if method == 'ST':
    result_train = joblib.Parallel(n_jobs=njobs,backend='threading')(joblib.delayed(state_tomography)(circ, nshots, backend, backend_qiskit) for circ in qibo_training_circuits)
    result_val = joblib.Parallel(n_jobs=njobs,backend='threading')(joblib.delayed(state_tomography)(circ, nshots, backend, backend_qiskit) for circ in qibo_validation_circuits)
elif method == 'CS':
    result_train = joblib.Parallel(n_jobs=njobs,backend='threading')(joblib.delayed(classical_shadows)(circ, shadow_size, backend, backend_qiskit) for circ in qibo_training_circuits)
    result_val = joblib.Parallel(n_jobs=njobs,backend='threading')(joblib.delayed(classical_shadows)(circ, shadow_size, backend, backend_qiskit) for circ in qibo_validation_circuits)
elif method == 'ST_qiskit':
    result_train = joblib.Parallel(n_jobs=njobs,backend='threading')(joblib.delayed(qiskit_state_tomography)(circ, nshots, backend_qiskit) for circ in qibo_training_circuits)
    result_val = joblib.Parallel(n_jobs=njobs,backend='threading')(joblib.delayed(qiskit_state_tomography)(circ, nshots, backend_qiskit) for circ in qibo_validation_circuits)

np.save(benchmark_circ_path+'/density_matrices_train.npy',result_train)
np.save(benchmark_circ_path+'/density_matrices_val.npy',result_val)