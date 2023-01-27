import itertools, json, random
import numpy as np
from qibo import gates
from qibo.backends import NumpyBackend
from qibo.models import Circuit


class Dataset(object):

    def __init__(self, n_circuits, n_gates, n_qubits):
        '''Generate dataset for the training of RL-algorithm

        Args:
            n_circuits (int): number of random circuits generated
            n_gates (int): number of gates in each circuit
            n_qubits (int): number of qubits in each circuit
        '''  

        self.n_gates=n_gates
        self.n_qubits=n_qubits
        self.circuits = [
            self.generate_random_circuit(nqubits=n_qubits, ngates=n_gates)
            for i in range(n_circuits)
        ]

    def get_train_loader(self):
        ''' Returns training set circuits'''
        return (c for c in self.train_circuits)

    def get_val_loader(self):
        ''' Returns validation set circuits'''
        return (c for c in self.val_circuits)

    def get_noisy_circuits(self):
        ''' Returns all noisy circuits'''
        return (c for c in self.noisy_circuits)

    def get_circuits(self):
        ''' Returns all circuits'''
        return (c for c in self.circuits)

    def __len__(self):
        return len(self.circuits)

    def __getitem__(self, i):
        return self.circuits[i]

    def noisy_shots(self, n_shots=1024, add_measurements=True, probabilities=False):
        '''Computes shots on noisy circuits

        Args:
            n_shots (int): number of shots executed
            add_measurements (bool): add measurement gates at the end of noisy circuits, necessary when this function is executed the first time
            probabilities (bool): normalize shot counts to obtain brobabilities

        Returns:
            shots_regiter (numpy.ndarray): number of shot counts/probabilities for each circuit in computational basis (order: 000; 001; 010 ...)
        '''
        if add_measurements:
            for i in range(len(self.circuits)):
                self.noisy_circuits[i].add(gates.M(*range(self.n_qubits)))
        shots_register_raw = [   
            self.noisy_circuits[i](nshots=n_shots).frequencies(binary=False)
            for i in range(len(self.circuits))
        ]
        shots_register=[]
        for i in range(len(self.circuits)):
            single_register=tuple(int(shots_register_raw[i][key]) for key in range(2**self.n_qubits))
            shots_register.append(single_register)
        if probabilities:
            return np.asarray(shots_register, dtype=float)/float(n_shots)
        else:
            return np.asarray(shots_register)

    def train_val_split(self, split=0.2, noise=False):
        '''Split dataset into train ad validation sets'''

        idx = random.sample(range(len(self.circuits)), int(split*len(self.circuits)))
        if not noise:
            self.val_circuits = [ c for i, c in enumerate(self.circuits) if i in idx ]
            self.train_circuits = [ c for i, c in enumerate(self.circuits) if i not in idx ]
        else:
            self.val_circuits = [ c for i, c in enumerate(self.noisy_circuits) if i in idx ]
            self.train_circuits = [ c for i, c in enumerate(self.noisy_circuits) if i not in idx ]

    def add_noise(self, noise_model='depolarising', noisy_gates=['rx'], noise_params=0.01):
        '''Add noise model to circuits

        Args:
            noise_model (str): noise model ("depolarising" or "depolarising_prop")
            noisy_gates (list): gates after which noise channels are added 
            noise_params (float): depolarising error parameter
        '''
        if noise_model=='depolarising':
            self.noisy_circuits = [
                self.add_dep_on_circuit(self.circuits[i], noisy_gates, noise_params)
                for i in range(len(self.circuits))
            ]
        # Depolarising parameter is proportional to rotation angle
        if noise_model=='depolarising_prop':
            self.noisy_circuits = [
                self.add_dep_on_circuit(self.circuits[i], noisy_gates, noise_params, prop=True)
                for i in range(len(self.circuits))
            ]

    def add_dep_on_circuit(self, circuit, noisy_gates, depolarizing_error, prop=False):
        '''Add noise model on a single circuit'''

        noisy_circ = Circuit(circuit.nqubits, density_matrix=True)
        time_steps = self.circuit_depth(circuit)
        for t in range(time_steps):
            for qubit in range(circuit.nqubits):
                gate = circuit.queue.moments[t][qubit]
                if gate == None:
                    pass
                elif gate.name in noisy_gates:
                    noisy_circ.add(circuit.queue.moments[t][qubit])
                    if not prop:
                        noisy_circ.add(
                            gates.DepolarizingChannel(
                                circuit.queue.moments[t][qubit].qubits, depolarizing_error
                            )
                        )
                    else:
                        noisy_circ.add(
                            gates.DepolarizingChannel(
                                circuit.queue.moments[t][qubit].qubits, depolarizing_error*gate.parameters[0]
                            )
                        )
                else:
                    noisy_circ.add(circuit.queue.moments[t][qubit])
        return noisy_circ

    def generate_dataset_representation(self):
        '''Generate dataset with features representing the circuit to be used as input for the RL algorithm. 
            For 1q circuits the feature representation is of dim (n_circuits, n_gates, 2).
            For multi qubit circuits the feature representation is of dim (n_circuits, max_circuit_moments, n_qubits, 3).
            More info in generate_circuit_representation.

        Returns:
            self.representation (numpy.ndarray): circuits representation as a feature vector
        '''

        # This variable is useful to have circuit representations of the same size for multi qubit circuits
        max_depth = max([
                self.circuit_depth(self.__getitem__(i))
                for i in range(len(self.circuits))
            ])
        self.representation = np.asarray([
                self.generate_circuit_representation(self.__getitem__(i), max_depth)
                for i in range(len(self.circuits))
            ])
        return self.representation

    def generate_circuit_representation(self, circuit, max_depth):
        '''Generate feature representation vector for a single circuit.
            Features are organised as follows.
            1q circuits: 
                circuit_repr[t,0] gate name (1=rx; 0=rz)
                circuit_repr[t,1] rotation angle normalized
            multi qubit circuits:
                circuit_repr[t,qubit,0] gate type (1=1q; -1=2q; 0=no_gate)
                circuit_repr[t,qubit,1] gate name (1=rx; 0=rz)
                circuit_repr[t,qubit,2] rotatoin angle nomalized
        '''

        time_steps = self.circuit_depth(circuit)
        if circuit.nqubits == 1:
            circuit_repr=np.zeros((time_steps, 2), dtype=float)
            for t in range(time_steps):
                gate = circuit.queue.moments[t][0]
                if gate.name == 'rx':
                    circuit_repr[t,0]=1
                    circuit_repr[t,1]=gate.parameters[0]/(2*np.pi)
                else:
                    circuit_repr[t,1]=gate.parameters[0]/(2*np.pi)
        else:
            circuit_repr=np.zeros((max_depth, circuit.nqubits, 3), dtype=float)
            for t in range(time_steps):
                for qubit in range(circuit.nqubits):
                    gate = circuit.queue.moments[t][qubit]
                    if gate == None:
                        pass
                    elif len(gate.qubits) == 1:
                        circuit_repr[t,qubit,0]=1
                        if gate.name == 'rx':
                            circuit_repr[t,qubit,1]=1
                            circuit_repr[t,qubit,2]=gate.parameters[0]/(2*np.pi)
                        else:
                            circuit_repr[t,qubit,2]=gate.parameters[0]/(2*np.pi)
                    else:
                        circuit_repr[t,qubit,0]=-1
        return np.asarray(circuit_repr)

    def circuit_depth(self, circuit):
        """Returns the depth of a circuit (number of circuit moments)"""
        return max(circuit.queue.moment_index)

    @staticmethod
    def generate_random_circuit(nqubits, ngates):
        """Generate a random circuit with RX, RZ and CZ gates."""
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
    
    def save_circuits(self, filename):
        '''Save circuits in json file'''
        circ_dict = {}
        for i, circ in enumerate(self.circuits):
            gate_list = []
            for gate in circ.queue:
                if len(gate.qubits)==1:
                    gate_list.append({
                        'name': gate.name.upper(), 
                        'qubit': gate.qubits,
                        'kwargs': gate.init_kwargs
                    })
                else:
                    gate_list.append({
                        'name': gate.name.upper(), 
                        'qubit': (int(gate.qubits[0]), int(gate.qubits[1])),
                        'kwargs': gate.init_kwargs
                    })
            circ_dict[i] = gate_list
        with open(filename, 'w') as f:
            json.dump(circ_dict, f, indent=2)

    def load_circuits(self, filename):
        '''Load circuits from json file'''
        self.circuits = []
        with open(filename, 'r') as f:
            circuits = json.load(f)
        for gate_list in circuits.values():
            nqubits = len(set(itertools.chain(*[g['qubit'] for g in gate_list])))
            circ = Circuit(nqubits)
            for g in gate_list:
                if len(g['qubit']) == 1:
                    circ.add(getattr(gates, g['name'])(g['qubit'][0], **g['kwargs']))
                else:
                    circ.add(getattr(gates, g['name'])(g['qubit'][0], g['qubit'][1], **g['kwargs']))
            self.circuits.append(circ)


