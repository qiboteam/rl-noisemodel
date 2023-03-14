import itertools, json, random
import numpy as np
from qibo import gates
from qibo.models import Circuit
from copy import deepcopy
from inspect import signature


class Dataset(object):

    def __init__(self, n_circuits, n_gates, n_qubits, primitive_gates=None, noise_model=None, mode='rep'):
        '''Generate dataset for the training of RL-algorithm

        Args:
            n_circuits (int): number of random circuits generated
            n_gates (int): number of gates in each circuit
            n_qubits (int): number of qubits in each circuit
        '''  

        self.n_gates = n_gates
        self.n_qubits = n_qubits
        self.noise_model = noise_model
        if primitive_gates is None:
            primitive_gates = ['RX', 'RZ', 'CZ'] if n_qubits > 1 else ['RX', 'RZ']
        self.primitive_gates = [ getattr(gates, g) for g in primitive_gates ]
        self.mode = mode
        
        self.circuits = [
            self.generate_random_circuit()
            for i in range(n_circuits)
        ]
        self.circ_rep = np.asarray([
            self.circuit_to_rep(c)
            for c in self.circuits
        ])
        self.train_circuits, self.val_circuits = self.train_val_split()

    def generate_random_circuit(self):
        """Generate a random circuit."""
        circuit = Circuit(self.n_qubits)
        for _ in range(self.n_gates):
            for q0 in range(self.n_qubits):
                gate = random.choice(self.primitive_gates) # maybe add identity gate?
                params = signature(gate).parameters
                if 'q0' in params and 'q1' in params:
                    q1 = random.choice(
                        list( set(range(self.n_qubits)) - {q0} )
                    )
                    circuit.add(gate(q1,q0))
                else:
                    if issubclass(gate, gates.ParametrizedGate):
                        theta = 2 * np.pi * np.random.random()
                        circuit.add(gate(q0, theta=theta))
                    else:
                        circuit.add(gate(q0))
        if self.noise_model is not None:
            self.noise_model.apply(circuit)
        return circuit

    def circuit_to_rep(self, circuit):
        rep = np.zeros((self.n_gates, 2))
        for i,gate in enumerate(circuit.queue):
            if isinstance(gate, gates.RZ):
                rep[i][1] = gate.init_kwargs["theta"]
            elif isinstance(gate, gates.RX):
                rep[i][0] = 1
                rep[i][1] = gate.init_kwargs["theta"]
        return rep

    def train_val_split(self, split=0.2):
        '''Split dataset into train ad validation sets'''

        idx = random.sample(range(len(self.circuits)), int(split*len(self.circuits)))
        val_circuits = [ c for i, c in enumerate(self.circuits) if i in idx ]
        train_circuits = [ c for i, c in enumerate(self.circuits) if i not in idx ]
        return train_circuits, val_circuits
        
    def get_train_loader(self):
        ''' Returns training set circuits'''
        return (c for c in self.train_circuits)

    def get_val_loader(self):
        ''' Returns validation set circuits'''
        return (c for c in self.val_circuits)

    def get_circuits(self):
        ''' Returns all circuits'''
        return (c for c in self.circuits)

    def __len__(self):
        return len(self.circuits)

    def set_mode(self, mode):
        self.mode = mode
    
    def __getitem__(self, i):
        if self.mode == 'rep':
            return self.circ_rep[i]
        elif self.mode == 'circ':
            return self.circuits[i]

    def pauli_probabilities(self, observable='Z', n_shots=100, n_rounds=100):
        '''Computes the probability distibutions of Pauli observables for 1q noisy circuits

        Args:
            oservable (string): pauli observable
            n_shots (int): number of shots executed for one observable estimation
            n_rounds (int): number of estimations of one observable

        Returns:
            register (np.ndarray): average observable value obtained at each measurement round
        '''

        circuits = deepcopy(self.noisy_circuits)
        self.add_masurement_gates(circuits, observable=observable)
        register=np.ndarray((len(self.circuits), n_rounds), dtype=float)
        for i in range(n_rounds):
            probs=self.compute_shots(circuits, n_shots=n_shots, probabilities=True)
            register[:,i]=probs[:,0]-probs[:,1]
        return register

    def generate_labels(self, n_shots=100, n_rounds=100):
        '''Generate the labels, containing the first two moments of distributions, necessary for training
        Args:
            oservable (string): pauli observable
            n_shots (int): number of shots executed for one observable estimation
            n_rounds (int): number of estimations of one observable

        Returns:
            moments (np.ndarray): array containing the first two moments of the distribution for each observable
        '''
        moments=np.ndarray((len(self.circuits), 3, 2), dtype=float)
        n_obs=0
        for obs in ['Z', 'Y', 'X']:
            register=self.pauli_probabilities(observable=obs, n_shots=n_shots, n_rounds=n_rounds)
            moments[:,n_obs,0]=np.mean(register, axis=1)
            moments[:,n_obs,1]=np.var(register, axis=1)
            n_obs+=1
        return moments
    
    def add_masurement_gates(self, circuits, observable='Z'):
        '''Add measurement gates at the end of circuits dataset, necessary when circuits are executed the first time
        
        Args: 
            circuits: circuits dataset
            observable (string): Pauli observable to be measured ('Z' for computational basis)
        '''

        for i in range(len(self.circuits)):
            if observable=='X' or observable=='Y':
                circuits[i].add(gates.H(*range(self.n_qubits)))
            if observable=='Y':
                circuits[i].add(gates.SDG(*range(self.n_qubits)))
            circuits[i].add(gates.M(*range(self.n_qubits)))
        

    def compute_shots(self, circuits, n_shots=1024, probabilities=False):
        '''Computes shots of circuits dataset

        Args:
            circuits: circuits dataset 
            n_shots (int): number of shots executed
            probabilities (bool): normalize shot counts to obtain probabilities

        Returns:
            shots_regiter (numpy.ndarray): number of shot counts/probabilities for each circuit in computational basis (order: 000; 001; 010 ...)
        '''

        shots_register_raw = [   
            circuits[i](nshots=n_shots).frequencies(binary=False)
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
                        # TODO: find 2 qubit gates matching for circuits with more than 3q
                        circuit_repr[t,qubit,0]=-1
        return np.asarray(circuit_repr)

    def circuit_depth(self, circuit):
        """Returns the depth of a circuit (number of circuit moments)"""
        return max(circuit.queue.moment_index)
    
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


    def repr_to_circuit(self, repr, noise_channel):
        c = Circuit(1, density_matrix=True)
        for gate, angle, noise_c, _ in repr[0]:
            if gate == 0:
                c.add(gates.RZ(
                    0,
                    theta = angle*2*np.pi,
                    trainable = False
                ))
            else:
                c.add(gates.RX(
                    0,
                    theta = angle*2*np.pi,
                    trainable = False
                ))
            if noise_c == 1:
                c.add(noise_channel)
        return c
