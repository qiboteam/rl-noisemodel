import itertools, json, random
import numpy as np
from qibo import gates
from qibo.backends import NumpyBackend
from qibo.models import Circuit


class Dataset(object):

    def __init__(self, n_circuits, n_gates, n_qubits, val_split=0.2, noise_model=None):
        self.noise_model = noise_model
        self.circuits = [
            self.generate_random_circuit(nqubits=n_qubits, ngates=n_gates)
            for i in range(n_circuits)
        ]
        self.train_val_split(val_split)

    def __len__(self):
        return len(self.circuits)

    def __getitem__(self, i):
        return self.circuits[i]

    def train_val_split(self, split=0.2):
        idx = random.sample(range(len(self.circuits)), int(split*len(self.circuits)))
        self.val_circuits = [ c for i, c in enumerate(self.circuits) if i in idx ]
        self.train_circuits = [ c for i, c in enumerate(self.circuits) if i not in idx ]

    @staticmethod
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

    def get_train_loader(self):
        return (c for c in self.train_circuits)

    def get_val_loader(self):
        return (c for c in self.val_circuits)
    
    def save_circuits(self, filename):
        circ_dict = {}
        for i, circ in enumerate(self.circuits):
            gate_list = []
            for gate in circ.queue:
                gate_list.append({
                    'name': gate.name.upper(), 
                    'qubit': gate.qubits,
                    'kwargs': gate.init_kwargs
                })
            circ_dict[i] = gate_list
        with open(filename, 'w') as f:
            json.dump(circ_dict, f, indent=2)

    def load_circuits(self, filename):
        self.circuits = []
        with open(filename, 'r') as f:
            circuits = json.load(f)
        for gate_list in circuits.values():
            nqubits = len(set(itertools.chain(*[g['qubit'] for g in gate_list])))
            circ = Circuit(nqubits)
            for g in gate_list:
                circ.add(getattr(gates, g['name'])(g['qubit'][0], **g['kwargs']))
            self.circuits.append(circ)


