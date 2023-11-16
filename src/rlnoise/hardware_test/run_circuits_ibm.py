import copy
import os

import joblib
import numpy as np
from qibo import gates
from qibo.backends import construct_backend
from qiskit.providers.fake_provider import FakeHanoiV2

from rlnoise.dataset import CircuitRepresentation
from rlnoise.hardware_test.utils import classical_shadows, qiskit_state_tomography, state_tomography

benchmark_circ_path="src/rlnoise/hardware_test/train_datasets/"
bench_name='IBM_train_D15_3Q_len500.npz'
bench_results_path = "src/rlnoise/hardware_test/dm_3Q_IBM/Hanoi_optimized_and_fixed/"
f = open(benchmark_circ_path+bench_name,"rb")
tmp=np.load(f,allow_pickle=True)

train_set=copy.deepcopy(tmp['train_set'])
val_set=copy.deepcopy(tmp['val_set'])
print(val_set.shape)
print(train_set.shape)

#Those lists contain 400 and 100 qibo circuits respectively 
qibo_training_circuits=[CircuitRepresentation().rep_to_circuit(train_set[i]) for i in range(train_set.shape[0])]
qibo_validation_circuits=[CircuitRepresentation().rep_to_circuit(val_set[i]) for i in range(val_set.shape[0])]
qibo_circ_all = qibo_training_circuits + qibo_validation_circuits

#IBMProvider.save_account(token='b957a8a1f5c7c3dcc553b397b1619f577b96187a10582d2467d3ebdc8f8351830e4a7a3148f7c7e54bffe17fc07ab73bdf7a2f45eacc649964a9f94db0d0f958')

qiskit = True
nshots = 1000
shadow_size = 100
method = 'ST'
likelihood = True
njobs = 2 #Only for 'CS' and 'ST_qiskit'
backend_qibo = 'tii1q_b1'
# backend_qiskit = 'ibm_hanoi'
backend_qiskit = "ibm_hanoi"
layout = [3, 5, 8]

if qiskit or method=='ST_qiskit':
    from qiskit_ibm_provider import IBMProvider
    provider = IBMProvider()
    backend = construct_backend('numpy')
    backend_qiskit = provider.get_backend(backend_qiskit)
else:
    from rlnoise.hardware_test.utils import rx_rule, x_rule
    os.environ["QIBOLAB_PLATFORMS"] = "/nfs/users/alejandro.sopena/qibolab_platforms_qrc/" #PATH to runcards
    backend = construct_backend('qibolab',backend_qibo)
    backend.compiler.__setitem__(gates.RX, rx_rule)
    backend.compiler.__setitem__(gates.X, x_rule)
    backend.transpiler = None
    backend_qiskit = None

batch_size = 4
for idx in range(348,len(qibo_circ_all),batch_size):
    if method == 'ST':
        results = state_tomography(qibo_circ_all[idx:idx+batch_size], nshots, likelihood, backend, backend_qiskit, layout)
        #result_train = results[0:len(qibo_training_circuits)]
        #result_val = results[len(qibo_training_circuits):]
    elif method == 'CS':
        result_train = joblib.Parallel(n_jobs=njobs,backend='threading')(joblib.delayed(classical_shadows)(circ, shadow_size, backend, backend_qiskit) for circ in qibo_training_circuits)
        result_val = joblib.Parallel(n_jobs=njobs,backend='threading')(joblib.delayed(classical_shadows)(circ, shadow_size, backend, backend_qiskit) for circ in qibo_validation_circuits)
    elif method == 'ST_qiskit':
        for circ in qibo_validation_circuits:
            qibo_training_circuits.append(circ)
        batch_size = 1
        for i in range(348,500, batch_size):   
            batch_result = joblib.Parallel(n_jobs=1,backend='threading')(joblib.delayed(qiskit_state_tomography)(circ, nshots, backend_qiskit) for circ in qibo_training_circuits[i:i+batch_size])
            
            np.save("src/rlnoise/hardware_test/dm_3Q_IBM/hanoi_fixed"+'/3Q_density_matrices_val_hanoi_ST_qiskit_%d.npy'%(i), np.array(batch_result, dtype=object))

    np.save(bench_results_path+'density_matrices_'+ str(idx)+'.npy',np.array(results, dtype=object))
    #np.save(benchmark_circ_path+'/density_matrices_val.npy',result_val)


