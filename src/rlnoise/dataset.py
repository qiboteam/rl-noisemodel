import random
import json
import os
import copy
import numpy as np
from qibo import gates
from qibo.quantum_info.random_ensembles import random_clifford
from qibo.models import Circuit
from rlnoise.noise_model import string_to_gate, CustomNoiseModel
from rlnoise.circuit_representation import CircuitRepresentation

def load_dataset(filename):
    '''Load a dataset from a npz file.
    Returns the circuits and labels.
    '''

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")
    
    tmp = np.load(filename, allow_pickle=True)
    circuits = copy.deepcopy(tmp['circuits'])
    labels = copy.deepcopy(tmp['labels'])
    return circuits, labels

class Dataset(object):
    def __init__(self, config_file):
        '''
        Generate dataset for the training of RL-algorithm.
        '''
        super(Dataset, self).__init__()
        with open(config_file) as f:
            config = json.load(f)
        
        self.primitive_gates = config['noise']['primitive_gates']
        dataset_options = config['dataset']
        self.n_gates = dataset_options['moments']
        self.n_qubits = dataset_options['qubits']
        self.n_circuits = dataset_options['n_circuits']
        self.clifford = dataset_options['clifford']
        enhanced_dataset = dataset_options['enhanced']
        rep = CircuitRepresentation(config_file)
        noise_model = CustomNoiseModel(config_file)
        if enhanced_dataset:
            self.circuits = [self.generate_clifford_circuit() for _ in range(self.n_circuits)]
        else:
            self.circuits = [self.generate_random_circuit() for _ in range(self.n_circuits)]
        self.noisy_circuits = [noise_model.apply(c) for c in self.circuits]
        self.dm_labels = np.asarray([self.noisy_circuits[i]().state() for i in range(self.n_circuits)])
        self.circ_rep = np.asarray([rep.circuit_to_array(c)for c in self.circuits], dtype=object)

    def generate_clifford_circuit(self):
        '''Generate a random Clifford circuit'''
        circuit = random_clifford(self.n_qubits, return_circuit=True, density_matrix=True)
        new_circuit = Circuit(self.n_qubits, density_matrix=True)
        for gate in circuit.queue:
            if gate.name.upper() in self.primitive_gates:
                new_circuit.add(gate)
            elif gate.name == "cx":
                new_circuit.add(gates.RZ(gate.qubits[1], np.pi/2))
                new_circuit.add(gates.RX(gate.qubits[1], np.pi/2))
                new_circuit.add(gates.CZ(gate.qubits[0], gate.qubits[1]))
                new_circuit.add(gates.RZ(gate.qubits[1], np.pi/2))
                new_circuit.add(gates.RX(gate.qubits[1], np.pi/2))
            elif gate.name == "h":
                new_circuit.add(gates.RZ(gate.qubits[0], np.pi/2))
                new_circuit.add(gates.RX(gate.qubits[0], np.pi/2))
            elif gate.name == "z":
                new_circuit.add(gates.RZ(gate.qubits[0], np.pi))
            elif gate.name == "y":
                new_circuit.add(gates.RZ(gate.qubits[0], np.pi))
                new_circuit.add(gates.RX(gate.qubits[0], np.pi))
            elif gate.name == "x":
                new_circuit.add(gates.RX(gate.qubits[0], np.pi))
            elif gate.name == "s":
                new_circuit.add(gates.RZ(gate.qubits[0], np.pi/2))
            else:
                raise ValueError(f"Unknown gate {gate.name}")
        return new_circuit

    def generate_random_circuit(self):
        """Generate a random circuit."""
        if self.n_qubits < 2 and "CZ" in self.primitive_gates:
            raise ValueError("Impossible to use CZ on single qubit circuits.")
        circuit = Circuit(self.n_qubits, density_matrix=True)
        while len(circuit.queue) < self.n_gates:
            q0 = random.choice(range(self.n_qubits))
            gate = string_to_gate(random.choice(self.primitive_gates))
            if isinstance(gate, gates.CZ):
                q1 = random.choice(
                    list(set(range(self.n_qubits)) - {q0})
                )       
                circuit.add(gate(q1,q0))
            elif issubclass(gate, gates.ParametrizedGate):
                theta = (
                    random.choice([0, 0.25, 0.5, 0.75])
                    if self.clifford
                    else np.random.random()
                )
                theta *= 2 * np.pi
                circuit.add(gate(q0, theta=theta))
        return circuit
    
    def save(self, filename):
        '''Save the dataset to a npz file'''
        
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        np.savez(filename,
                circuits = self.circ_rep,
                labels = self.dm_labels,
                allow_pickle=True)