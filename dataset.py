import itertools, json
import numpy as np
from qibo import gates
from qibo.backends import NumpyBackend
from qibo.models import Circuit

from qibolab.tests.test_transpilers_connectivity import transpose_qubits
from qibolab.transpilers.transpile import can_execute, transpile


def generate_random_circuit(nqubits, ngates):
    """Generate random circuits one-qubit rotations and CZ gates."""
    pairs = list(itertools.combinations(range(nqubits), 2))

    one_qubit_gates = [gates.RX, gates.RZ]
    two_qubit_gates = [gates.CZ]
    n1, n2 = len(one_qubit_gates), len(two_qubit_gates)
    n = n1 + n2 if nqubits > 1 else n1
    circuit = Circuit(nqubits)
    for _ in range(ngates):
        igate = int(np.random.randint(0, n))
        if igate >= n1:
            q = tuple(np.random.randint(0, nqubits, 2))
            while q[0] == q[1]:
                q = tuple(np.random.randint(0, nqubits, 2))
            gate = two_qubit_gates[igate - n1]
        else:
            q = (np.random.randint(0, nqubits),)
            gate = one_qubit_gates[igate]
        if issubclass(gate, gates.ParametrizedGate):
            theta = 2 * np.pi * np.random.random()
            circuit.add(gate(*q, theta=theta))
        else:
            circuit.add(gate(*q))
    return circuit

def save_circuit(circ):
    gate_list = []
    for gate in circ.queue:
        gate_list.append({
            'name': gate.name.upper(), 
            'qubit': gate.qubits,
            'kwargs': gate.init_kwargs
        })
    return gate_list

def load_circuit(gate_list, nqubits):
    circ = Circuit(nqubits=nqubits)
    for g in gate_list:
        circ.add(getattr(gates, g['name'])(g['qubit'][0], **g['kwargs']))
    return circ


nqubits = 1
ngates = 10
ncirc = 5
dataset = {}

# create dataset
for i in range(ncirc):
    circ = generate_random_circuit(nqubits, ngates)
    dataset[i] = save_circuit(circ)
    
with open('dataset.json', 'w') as f:
    json.dump(dataset, f, indent=2)

with open('dataset.json', 'r') as f:
    dataset = json.load(f)

#for v in dataset.values():
#    print(load_circuit(v, nqubits=nqubits).draw())

