import os
import numpy as np
import copy
from qibo import set_backend, gates
from qibo.config import log
from qibolab.compilers.compiler import Compiler
from qibo.backends import construct_backend
from rlnoise.dataset import CircuitRepresentation
from rlnoise.rewards.state_tomography import StateTomography
import math
from qibolab.pulses import PulseSequence


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

os.environ["QIBOLAB_PLATFORMS"] = "/nfs/users/alejandro.sopena/qibolab_platforms_qrc/" #PATH to runcards
qibolab = construct_backend('qibolab','tii1q_b1')
numpy = construct_backend('numpy')

def rx_rule(gate, platform):

    num = int(gate.parameters[0] / (np.pi/2))
    start = 0
    sequence = PulseSequence()
    for _ in range(num):
        qubit = gate.target_qubits[0]
        RX90_pulse = platform.create_RX90_pulse(
            qubit,
            start=start,
            relative_phase=0,
        )
        sequence.add(RX90_pulse)
        start = RX90_pulse.finish

    return sequence, {}

def x_rule(gate, platform):

    num = 2
    start = 0
    sequence = PulseSequence()
    for _ in range(num):
        qubit = gate.target_qubits[0]
        RX90_pulse = platform.create_RX90_pulse(
            qubit,
            start=start,
            relative_phase=0,
        )
        sequence.add(RX90_pulse)
        start = RX90_pulse.finish

    return sequence, {}

qibolab.compiler.__setitem__(gates.RX, rx_rule)
qibolab.compiler.__setitem__(gates.X, x_rule)
qibolab.transpiler = None

rhos = []
for circ in qibo_training_circuits:

    st = StateTomography(backend=qibolab)

    st.get_circuits(circ)
    st.meas_obs(noise=None,readout_mit=False)
    rho = st.get_rho()

    st.meas_obs(noise=None,readout_mit=True)
    cal_mat = st.cal_mat
    rho_mit = st.get_rho()

    circ.density_matrix = True
    rho_exact = numpy.execute_circuit(circ).state()
    log.info(circ.draw())
    log.info([circ, rho_exact, rho, rho_mit, cal_mat])
    rhos.append([circ, rho_exact, rho, rho_mit, cal_mat])


np.save(benchmark_circ_path+'density_matrices.npy',rhos)