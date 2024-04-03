import os
import numpy as np
from qibo import gates
from qibo.backends import construct_backend
from rlnoise.utils_hardware import state_tomography

benchmark_circ_path= "src/rlnoise/hardware_test/dm_1Q/"
bench_name='100_circ_set.npy'
bench_results_path = "src/rlnoise/hardware_test/dm_1Q/"

f = open(benchmark_circ_path+bench_name,"rb")
#Those lists contain 400 and 100 qibo circuits respectively 
qibo_circuits=np.load(f,allow_pickle=True)
print(len(qibo_circuits))

quantum_spain = True
nshots = 1000
shadow_size = 100
method = 'ST'
likelihood = False
njobs = 2 #Only for 'CS' and 'ST_qiskit'
backend_qibo = 'tii1q_b1'
backend_qs = 9

layout = [2]
if quantum_spain:
    backend_qs = backend_qs
    backend = construct_backend('numpy')
else:
    from rlnoise.utils_hardware import rx_rule, x_rule
    os.environ["QIBOLAB_PLATFORMS"] = "/nfs/users/alejandro.sopena/qibolab_platforms_qrc/" #PATH to runcards
    backend = construct_backend('qibolab',backend_qibo)
    backend.compiler.__setitem__(gates.RX, rx_rule)
    backend.compiler.__setitem__(gates.X, x_rule)
    backend.transpiler = None
    backend_qiskit = None
if method == 'ST':
    results = state_tomography(qibo_circuits, nshots, likelihood, backend, backend_qiskit, backend_qs, layout)
    
np.save(bench_results_path+bench_name[0:-4]+f'_result_qubit{layout[0]}_NEW.npy',np.array(results, dtype=object))
