from qibo import Circuit, gates
from qibo.backends import NumpyBackend
from qibolab.backends import QibolabBackend

circuit = Circuit(1)
circuit.add(gates.RZ(0, 0.1))
circuit.add(gates.M(0))

#backend = NumpyBackend()
backend = QibolabBackend(platform = 'qw11q')

result = backend.execute_circuit(circuit, nshots=100)
print(result)