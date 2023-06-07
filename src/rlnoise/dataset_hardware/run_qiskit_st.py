import os
import numpy as np
import copy
from qibo.config import log
from rlnoise.dataset import CircuitRepresentation

from qiskit_ibm_provider import IBMProvider
from qiskit import QuantumCircuit
from qiskit_experiments.library import MitigatedStateTomography, StateTomography
from qiskit.quantum_info import DensityMatrix
from qiskit_aer import StatevectorSimulator

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
provider = IBMProvider()

backend = provider.get_backend("ibmq_qasm_simulator")
backend_exact = StatevectorSimulator()

rhos = []
for circ in qibo_training_circuits:
    circ = qibo_training_circuits[0]

    circ_qiskit = QuantumCircuit().from_qasm_str(circ.to_qasm())


    st = StateTomography(circ_qiskit,backend=backend)
    st.set_transpile_options(optimization_level=0)
    results = st.run(backend, seed_simulation=100)
    rho = results.analysis_results("state").value.to_operator().data


    st = MitigatedStateTomography(circ_qiskit,backend=backend)
    st.set_transpile_options(optimization_level=0)
    results = st.run(backend)
    rho_mit = results.analysis_results("state").value.to_operator().data

    cal_mat = results.analysis_results("Local Readout Mitigator").value.assignment_matrix()

    state_exact = backend_exact.run(circ_qiskit).result().get_statevector()
    rho_exact = DensityMatrix(state_exact).to_operator().data


    log.info(circ.draw())
    log.info([circ, rho_exact, rho, rho_mit, cal_mat])
    rhos.append([circ, rho_exact, rho, rho_mit, cal_mat])


np.save(benchmark_circ_path+'density_matrices_st.npy',rhos)
