import itertools, json, random
import numpy as np
from qibo import gates
from qibo.models import Circuit
from copy import deepcopy
from inspect import signature


class Dataset(object):

    def __init__(self, n_circuits, n_gates, n_qubits, clifford=True, primitive_gates=None, noise_model=None, mode='rep'):
        '''Generate dataset for the training of RL-algorithm

        Args:
            n_circuits (int): number of random circuits generated
            n_gates (int): number of gates in each circuit (depth)
            n_qubits (int): number of qubits in each circuit
        '''

        super(Dataset, self).__init__()

        self.n_gates = n_gates
        self.n_qubits = n_qubits
        self.clifford = clifford
        self.noise_model = noise_model
        if primitive_gates is None:
            primitive_gates = {'RZ': 0, 'RX': 1, 'CZ': -1} if n_qubits > 1 else {'RZ': 0, 'RX': 1}
        self.index2gate = { i: getattr(gates, g) for g,i in primitive_gates.items() }
        self.gate2index = { getattr(gates, g): i for g,i in primitive_gates.items() }
        self.mode = mode
        
        self.circuits = [
            self.generate_random_circuit()
            for i in range(n_circuits)
        ]
        if self.noise_model is not None:
            self.noisy_circuits = [
                self.noise_model.apply(c)
                for c in self.circuits
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
                if self.clifford:
                    raise AssertionError("To be implemented.")
                gate = random.choice(list(self.index2gate.values())) # maybe add identity gate?
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
        return circuit

    def circuit_to_rep(self, circuit):
        """Maps qibo circuits to numpy array representation."""
        rep = np.zeros((self.n_gates, 2*self.n_qubits))
        # looping over moments
        for i,moment in enumerate(circuit.queue.moments):
            for j,gate in enumerate(moment):
                # checking whether it's a 2-qubits gate
                if len(gate.init_args) > 1:
                    raise AssertionError ("To be implemented.")
                else:
                    rep[i,2*j] = self.gate2index[type(gate)]
                    rep[i,2*j+1] = gate.init_kwargs["theta"] / (2 * np.pi)
        return rep

    def train_val_split(self, split=0.2):
        '''Split dataset into train ad validation sets'''

        idx = random.sample(range(len(self.circuits)), int(split*len(self.circuits)))
        val_circuits = [ self.__getitem__(i) for i in range(self.__len__()) if i in idx ]
        train_circuits = [ self.__getitem__(i) for i in range(self.__len__()) if i not in idx ]
        if self.mode == 'rep':
            val_circuits = np.asarray(val_circuits)
            train_circuits = np.asarray(train_circuits)
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
        elif self.mode == 'noisy_circ':
            return self.noisy_circuits[i]
    
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

    def rep_to_circuit(self, rep, noise_channel):
        c = Circuit(self.n_qubits, density_matrix=True)
        for *row, _ in rep[0]:
            gates = row[:self.n_qubits]
            angles = row[self.n_qubits:2*self.n_qubits]
            noise_ch = row[-self.n_qubits:]
            for i, (gate, angle, nc) in enumerate(zip(gates, angles, noise_ch)):
                gate = self.index2gate[gate]
                # check for 2-qubits gates
                params = signature(gate).parameters
                if 'q0' in params and 'q1' in params:
                    raise AssertionError("To be implemented.")
                else:
                    c.add(gate(
                        i,
                        theta = angle * 2 * np.pi,
                        trainable = False
                    ))
                if nc == 1:
                    if self.n_qubits > 1:
                        raise AssertionError("To be implemented.")
                    c.add(noise_channel) # this has to be adapted for multi-qubits
        return c


class CircuitRepresentation(object):

    def __init__(self, gates_map, noise_channels_map, shape='2d'):
        super(CircuitRepresentation, self).__init__()

        self.gate2index = { getattr(gates, k): v for k,v in gates_map.items() }
        self.index2gate = { v: k for k,v in self.gate2index.items() }
        self.channel2index = { getattr(gates, k): v for k,v in noise_channels_map.items() }
        self.index2channel = { v: k for k,v in self.channel2index.items() }
        self.shape = shape

    def circuit_to_array(self, circuit):
        """Maps qibo circuits to numpy array representation."""
        depth = len(circuit.queue.moments)
        nqubits = circuit.nqubits
        rep = np.zeros((
            depth,
            nqubits,
            len(self.gate2index) + len(self.channel2index)
        ))
        # looping over moments
        for i,moment in enumerate(circuit.queue.moments):
            for j,gate in enumerate(moment):
                # checking whether it's a 2-qubits gate
                if len(gate.init_args) > 1:
                    raise AssertionError ("To be implemented.")
                else:
                    try:
                        k = self.gate2index[type(gate)]
                        val = gate.init_kwargs["theta"] / (2 * np.pi)
                    except:
                        k = self.channel2index[type(gate)] + len(self.gate2index)
                        val = gate.init_kwargs["lam"]
                    rep[i, j, k] = val
        if self.shape == '2d':
            rep = rep.reshape(depth, -1)
        return rep

    def array_to_circuit(self, array):
        """Maps numpy array to qibo circuit."""
        depth = array.shape[0]
        if self.shape == '2d':
            array = array.reshape(depth, -1, len(self.gate2index) + len(self.channel2index))
        nqubits = array.shape[1]
        c = Circuit(nqubits, density_matrix=True)
        for moment in array:
            for qubit, row in enumerate(moment):
                # add the gate
                gate = row[:len(self.gate2index)]
                idx = int(gate.nonzero()[0])
                c.add(
                    self.index2gate[idx](
                        qubit,
                        theta = gate[idx]
                    ))
                # add the channel
                channel = row[-len(self.channel2index):]
                idx = channel.nonzero()[0]
                if len(idx) > 0:
                    idx = int(idx)
                    c.add(
                      self.index2channel[idx](
                          qubit,
                          lam = channel[idx]
                    ))
        return c
