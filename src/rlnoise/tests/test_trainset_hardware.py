import numpy as np
import os
from rlnoise.utils import test_avg_fidelity
from rlnoise.dataset import CircuitRepresentation
from rlnoise.metrics import compute_fidelity, bures_distance, trace_distance
from qiskit.quantum_info import DensityMatrix
from qibo.models.circuit import Circuit

#[circ, rho_exact, rho, rho_mit, st.cal_mat] 
f = open('src/rlnoise/hardware_test/dm_3Q_IBM/Hanoi_optimized_and_fixed/train_set_Hanoi_3Q.npy',"rb")
bench_name='src/rlnoise/hardware_test/train_datasets/IBM_train_D15_3Q_len500.npz'
f2 = open(bench_name, "rb")
dataset = np.load(f, allow_pickle=True)
original_set = np.load(f2, allow_pickle=True)
fidel = []
bures = []
trace_d = []
for idx, data in enumerate(dataset[:300]):
    #print(compute_fidelity(data[0]().state(),data[1])) #passed DMs are the same in qibo and qiskit
    #print(compute_fidelity(data[2],data[1])) #passed fidelity > 0.9
    #print(compute_fidelity(data[3],data[1])) #passed > 0.95
    fidel.append(compute_fidelity(data[3],data[1]))
    bures.append(bures_distance(data[3],data[1]))
    trace_d.append(trace_distance(data[3],data[1]))
fidel = np.array(fidel).mean()
bures = np.array(bures).mean()
trace_d = np.array(trace_d).mean()
print(fidel, bures, trace_d)
# original_circ_rep = original_set["train_set"]
# circ_qibo = [CircuitRepresentation().rep_to_circuit(circ) for circ in original_circ_rep]
# rep2 = [CircuitRepresentation().circuit_to_array(circ) for circ in circ_qibo]
# circ_qibo_2 = [CircuitRepresentation().rep_to_circuit(circ) for circ in rep2]
# #[print(compute_fidelity(circ_qibo[i]().state(), circ_qibo_2[i]().state())) for i in range(len(circ_qibo))]
# print(original_circ_rep[0].shape)
# [print(CircuitRepresentation().rep_to_circuit(circ).draw()) for circ in original_circ_rep]


f.close()
f2.close()

'''
circ_representation_train = []
labels_train = []
circ_representation_val = []
labels_val = []
exact_rho1 = []
exact_rho2 = []
results = []
batch_size = 4
for i in range(0,344, batch_size):
    f = open("src/rlnoise/hardware_test/dm_3Q_IBM/Hanoi_optimized_and_fixed/density_matrices_"+str(i)+".npy", "rb")
    tmp = np.load(f, allow_pickle=True)
    for data in tmp:
        results.append(data)
        #if compute_fidelity(data[2], data[0]().state()) > 0.5:
        
        #qiskit_circ = data[1].reverse_bits().qasm()
        #qibo_circ = Circuit(3, density_matrix=True).from_qasm(qiskit_circ)
        #circ_representation_train.append(CircuitRepresentation().circuit_to_array(qibo_circ))
        #labels_train.append(data[3])
        #exact_rho1.append(data[2])
        
        #print("fidelity: ", compute_fidelity(data[2], qibo3().state()))
            #print(compute_fidelity(data[2], data[0]().state()))
f2 = open("train_set_Hanoi_3Q.npy", "wb") 
#np.save(f2, np.array([circ_representation_train, labels_train, exact_rho1], dtype=object))
results = np.array(results, dtype=object)
np.save(f2, results)
f.close()
f2.close()
print("Results shape: ", results.shape)
'''
'''
for i in range(181,206):
    f = open("src/rlnoise/hardware_test/dm_3Q_IBM/hanoi2/3Q_density_matrices_val_hanoi_ST_qiskit_"+str(i)+".npy", "rb")
    tmp = np.load(f, allow_pickle=True)
    for data in tmp:
        
        qiskit_circ = data[1].reverse_bits().qasm()
        qibo_circ = Circuit(3, density_matrix=True).from_qasm(qiskit_circ)
        circ_representation_val.append(CircuitRepresentation().circuit_to_array(qibo_circ))
        labels_val.append(data[3])
        exact_rho2.append(data[2])
        

f2 = open("val_set_Hanoi_3Q.npy", "wb") 
np.save(f2, np.array([circ_representation_val, labels_val, exact_rho2], dtype=object))
f.close()
f2.close()
print("num circuits: ", len(circ_representation_val))
'''