from rlnoise.custom_noise import CustomNoiseModel
from qibo import gates
from qibo.models import Circuit


test_circuit=Circuit(3,density_matrix=True)
test_circuit.add(gates.RX(q=0,theta=0.5))
test_circuit.add(gates.RX(q=1,theta=0.5))
test_circuit.add(gates.RX(q=2,theta=0.5))
test_circuit.add(gates.RZ(q=0,theta=0.5))
test_circuit.add(gates.RZ(q=1,theta=0.5))
test_circuit.add(gates.RZ(q=2,theta=0.5))
test_circuit.add(gates.CZ(0,2))
print('Original test circuit: ')
print(test_circuit.draw())
noise=CustomNoiseModel()
noisy_circ=noise.apply(test_circuit)
print('Relative noisy circuit')
print(noisy_circ.draw())