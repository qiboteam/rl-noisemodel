import numpy as np
from qiskit.providers.fake_provider import FakeHanoiV2
from qiskit import transpile
from qiskit import QuantumCircuit
# Number of circuits
num_circuits = 400
nshots = 1000
original_circ_path = "src/rlnoise/hardware_test/dm_3Q_IBM/Hanoi_optimized_and_fixed/train_set_Hanoi_3Q.npy"
f = open(original_circ_path, "rb")
data = np.load(f, allow_pickle=True)
backend = FakeHanoiV2
# Create a list to store the circuits
circuits = []

# Create and add the circuits to the list
for i in range(len(data)):
    circ_qibo = data[i,0]
    qasm_circ = circ_qibo.to_qasm()
    qisk_circ = QuantumCircuit.from_qasm_str(qasm_circ)
    transpiled_circuit = transpile(qisk_circ, backend)
    job = backend.run(transpiled_circuit)
    print(job.result())



