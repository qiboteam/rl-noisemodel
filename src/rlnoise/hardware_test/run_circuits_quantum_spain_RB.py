import copy
import os

import joblib
import numpy as np
from qibo import gates
from qibo.backends import construct_backend
# from qiskit.providers.fake_provider import FakeHanoiV2

from rlnoise.dataset import CircuitRepresentation
from rlnoise.hardware_test.utils import classical_shadows, qiskit_state_tomography, state_tomography
from qibo.models import Circuit

benchmark_circ_path= "src/rlnoise/hardware_test/RB/" #"src/rlnoise/hardware_test/dm_1Q_quantum_spain/"


#benchmark_circ_path= "src/rlnoise/hardware_test/dm_1Q_quantum_spain/"
numbers = np.arange(1, 200, 20)
for number in numbers:
    bench_name='D'+str(number)+'_len20_qubit1.npy'#'100_circ_set.npy'

    bench_results_path = "src/rlnoise/hardware_test/RB/"#"src/rlnoise/hardware_test/dm_1Q_quantum_spain/"

    f = open(benchmark_circ_path+bench_name,"rb")


    #Those lists contain 400 and 100 qibo circuits respectively 
    qibo_circuits=np.load(f,allow_pickle=True)
    print(len(qibo_circuits))
    
    # qibo_circuits = qibo_circuits[0:100]

    #IBMProvider.save_account(token='b957a8a1f5c7c3dcc553b397b1619f577b96187a10582d2467d3ebdc8f8351830e4a7a3148f7c7e54bffe17fc07ab73bdf7a2f45eacc649964a9f94db0d0f958')

    qiskit = False
    quantum_spain = True
    nshots = 1000
    shadow_size = 100
    method = 'ST'
    likelihood = False
    njobs = 2 #Only for 'CS' and 'ST_qiskit'
    backend_qibo = 'tii1q_b1'
    # backend_qiskit = 'ibm_hanoi'
    backend_qiskit = "ibm_hanoi"
    backend_qiskit = None
    backend_qs = 9

    layout = [1]

    if qiskit or method=='ST_qiskit':
        from qiskit_ibm_provider import IBMProvider
        provider = IBMProvider()
        backend = construct_backend('numpy')
        backend_qiskit = provider.get_backend(backend_qiskit)
    elif quantum_spain:
        backend_qs = backend_qs
        backend = construct_backend('numpy')
    else:
        from rlnoise.hardware_test.utils import rx_rule, x_rule
        os.environ["QIBOLAB_PLATFORMS"] = "/nfs/users/alejandro.sopena/qibolab_platforms_qrc/" #PATH to runcards
        backend = construct_backend('qibolab',backend_qibo)
        backend.compiler.__setitem__(gates.RX, rx_rule)
        backend.compiler.__setitem__(gates.X, x_rule)
        backend.transpiler = None
        backend_qiskit = None





    if method == 'ST':
        results = state_tomography(qibo_circuits, nshots, likelihood, backend, backend_qiskit, backend_qs, layout)

    np.save(bench_results_path+bench_name[0:-4]+'_result.npy',np.array(results, dtype=object))