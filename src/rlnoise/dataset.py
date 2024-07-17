import random
import json
import os
import copy
import numpy as np
from qibo import gates
from qibo.quantum_info.random_ensembles import random_clifford
from qibo.models import Circuit
from rlnoise.noise_model import CustomNoiseModel
from rlnoise.circuit_representation import CircuitRepresentation
from rlnoise.utils_hardware import state_tomography

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
    def __init__(self, config_file, evaluation=False):
        '''
        Generate dataset for the training of RL-algorithm.
        '''
        super(Dataset, self).__init__()
        with open(config_file) as f:
            config = json.load(f)
        self.config = config
        
        self.primitive_gates = config['noise']['primitive_gates']
        dataset_options = config['dataset']
        self.n_gates = dataset_options['moments']
        self.n_qubits = dataset_options['qubits']
        self.n_circuits = dataset_options['n_circuits']
        self.clifford = dataset_options['clifford']
        self.eval_size = dataset_options['eval_size']
        self.eval_depth = dataset_options['eval_depth']
        enhanced_dataset = dataset_options['distributed_clifford']
        mixed = dataset_options['mixed']
        self.rep = CircuitRepresentation(config_file)
        self.noise_model = CustomNoiseModel(config_file)
        if enhanced_dataset and not mixed:
            print("Generating distributed clifford dataset.")
            self.circuits = [self.generate_clifford_circuit() for _ in range(self.n_circuits)]
        elif not mixed:
            print("Generating random dataset.")
            self.circuits = [self.generate_random_circuit() for _ in range(self.n_circuits)]
        elif mixed:
            print("Generating mixed dataset.")
            self.circuits = [self.generate_random_circuit() for _ in range(int(self.n_circuits/2))]
            self.circuits += [self.generate_clifford_circuit() for _ in range(int(self.n_circuits/2))]
            random.shuffle(self.circuits)
        else:
            raise ValueError("Unknown dataset type.")
        self.noisy_circuits = [self.noise_model.apply(c) for c in self.circuits]
        self.dm_labels = np.asarray([self.noisy_circuits[i]().state() for i in range(self.n_circuits)])
        self.circ_rep = np.asarray([self.rep.circuit_to_array(c)for c in self.circuits], dtype=object)

    def generate_rb_dataset(self, backend=None):
        rb_options = self.config["rb"]
        circuits_list = []
        labels = []
        for len in range(rb_options["start"], rb_options["stop"], rb_options["step"]):
            self.n_gates = len
            circuits = np.asarray([self.generate_random_circuit() for _ in range(rb_options["n_circ"])])
            circ_rep = [self.rep.circuit_to_array(c)for c in circuits]
            if backend is None or (backend.name != "QuantumSpain" and backend.name != "qibolab"):
                noisy_circuits = [self.noise_model.apply(c) for c in circuits]
                dm_labels = np.asarray([noisy_circuits[i]().state() for i in range(rb_options["n_circ"])])
            else:         
                nshots = self.config["chip_conf"]["nshots"]
                likelihood = self.config["chip_conf"]["likelihood"]
                readout_mitigation = self.config["chip_conf"]["readout_mitigation"]
                result = state_tomography(circuits, nshots, likelihood, backend)
                if readout_mitigation:
                    dm_labels = np.asarray([result[i][3] for i in range(rb_options["n_circ"])])
                else:
                    dm_labels = np.asarray([result[i][2] for i in range(rb_options["n_circ"])])
                cal_mat = result[0][4]
            labels.append(dm_labels)     
            circuits_list.append(circ_rep)
        circuits_list  = np.asarray(circuits_list, dtype=object)
        
        if backend is not None and (backend.name == "QuantumSpain" or backend.name == "qibolab"):
            np.savez(rb_options["dataset"], circuits = circuits_list, labels = labels, cal_mat = cal_mat)
        else:
            np.savez(rb_options["dataset"], circuits = circuits_list, labels = labels)

    def generate_eval_dataset(self, save_path, backend=None):
        '''Generate a dataset for evaluation of the RL-model'''
        self.n_gates = self.eval_depth
        circuits = [self.generate_random_circuit() for _ in range(self.eval_size)]
        circ_rep = np.asarray([self.rep.circuit_to_array(c)for c in circuits], dtype=object)
        if backend is not None and (backend.name == "QuantumSpain" or backend.name == "qibolab"):
            nshots = self.config["chip_conf"]["nshots"]
            likelihood = self.config["chip_conf"]["likelihood"]
            readout_mitigation = self.config["chip_conf"]["readout_mitigation"]
            result = state_tomography(circuits, nshots, likelihood, backend)
            if readout_mitigation:
                dm_labels = np.asarray([result[i][3] for i in range(self.eval_size)])
            else:
                dm_labels = np.asarray([result[i][2] for i in range(self.eval_size)])
            cal_mat = result[0][4]
            np.savez(save_path, circuits = circ_rep, labels = dm_labels, cal_mat = cal_mat)
        else:
            noisy_circuits = [self.noise_model.apply(c) for c in circuits]
            dm_labels = np.asarray([noisy_circuits[i]().state() for i in range(self.eval_size)])
            np.savez(save_path, circuits = circ_rep, labels = dm_labels)
        

    def generate_clifford_circuit(self):
        '''Generate a random Clifford circuit'''
        circuit = random_clifford(self.n_qubits, return_circuit=True, density_matrix=True)
        new_circuit = Circuit(self.n_qubits, density_matrix=True)
        for gate in circuit.queue:
            if gate.name in self.primitive_gates:
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

    def generate_smaller_circuits(self):
        """Generate a circuit with a smaller number of qubits to improve generalization properties."""
        qubits_subset = random.sample(range(self.n_qubits), random.randint(1, self.n_qubits-1))
        circuit = Circuit(self.n_qubits, density_matrix=True)
        while len(circuit.queue.moments) < self.n_gates:
            q0 = random.choice(qubits_subset)
            gate = random.choice(self.primitive_gates)
            if gate == 'cz':
                q1 = random.choice(
                    list(set(range(self.n_qubits)) - {q0})
                )       
                circuit.add(gates.CZ(q1,q0))
            elif gate == 'rx':
                theta = (
                    random.choice([0, 0.25, 0.5, 0.75])
                    if self.clifford
                    else np.random.random()
                )
                theta *= 2 * np.pi
                circuit.add(gates.RX(q0, theta=theta))
            elif gate == 'rz':
                theta = (
                    random.choice([0, 0.25, 0.5, 0.75])
                    if self.clifford
                    else np.random.random()
                )
                theta *= 2 * np.pi
                circuit.add(gates.RZ(q0, theta=theta))
            else:
                raise ValueError(f"Gate {gate} not present in the primitive gates.")
        return circuit

    def generate_random_circuit(self):
        """Generate a random circuit."""
        if self.n_qubits < 2 and "cz" in self.primitive_gates:
            raise ValueError("Impossible to use CZ on single qubit circuits.")
        circuit = Circuit(self.n_qubits, density_matrix=True)
        while len(circuit.queue.moments) < self.n_gates:
            q0 = random.choice(range(self.n_qubits))
            gate = random.choice(self.primitive_gates)
            if gate == 'cz':
                q1 = random.choice(
                    list(set(range(self.n_qubits)) - {q0})
                )       
                circuit.add(gates.CZ(q1,q0))
            elif gate == 'rx':
                theta = (
                    random.choice([0, 0.25, 0.5, 0.75])
                    if self.clifford
                    else np.random.random()
                )
                theta *= 2 * np.pi
                circuit.add(gates.RX(q0, theta=theta))
            elif gate == 'rz':
                theta = (
                    random.choice([0, 0.25, 0.5, 0.75])
                    if self.clifford
                    else np.random.random()
                )
                theta *= 2 * np.pi
                circuit.add(gates.RZ(q0, theta=theta))
            else:
                raise ValueError(f"Gate {gate} not present in the primitive gates.")
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