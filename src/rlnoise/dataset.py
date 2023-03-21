import itertools, json, random
import numpy as np
from qibo import gates
from qibo.models import Circuit
from inspect import signature


class Dataset(object):

    def __init__(self, n_circuits, n_gates, n_qubits, representation, clifford=True, noise_model=None, mode='rep'):
        '''Generate dataset for the training of RL-algorithm
        Args:
            n_circuits (int): number of random circuits generated
            n_gates (int): number of gates in each circuit (depth)
            n_qubits (int): number of qubits in each circuit
        '''

        super(Dataset, self).__init__()

        self.n_gates = n_gates
        self.n_qubits = n_qubits
        self.rep = representation
        self.clifford = clifford
        self.noise_model = noise_model
        self.mode = mode
        self.n_circuits=n_circuits
        
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
            self.rep.circuit_to_array(c)
            for c in self.circuits
        ])
        self.noisy_circ_rep = np.asarray([
            self.rep.circuit_to_array(c)
            for c in self.noisy_circuits
        ])
        self.train_circuits, self.val_circuits = self.train_val_split()

    def get_dm_labels(self):
        return np.asarray([self.noisy_circuits[i]().state()
                for i in range(self.n_circuits)])

    def generate_random_circuit(self):
        """Generate a random circuit."""
        circuit = Circuit(self.n_qubits, density_matrix=True)
        for _ in range(self.n_gates):
            for q0 in range(self.n_qubits):
                gate = random.choice(list(self.rep.index2gate.values())) # maybe add identity gate?
                params = signature(gate).parameters
                if 'q0' in params and 'q1' in params:
                    q1 = random.choice(
                        list( set(range(self.n_qubits)) - {q0} )
                    )
                    circuit.add(gate(q1,q0))
                else:
                    if issubclass(gate, gates.ParametrizedGate):
                        if self.clifford:
                            theta = random.choice([0, 0.25, 0.5, 0.75, 1])
                        else:
                            theta = np.random.random()
                        theta *= 2 * np.pi
                        circuit.add(gate(q0, theta=theta))
                    else:
                        circuit.add(gate(q0))
        return circuit

    def train_val_split(self, split=0.2):
        '''Split dataset into train ad validation sets'''

        idx = random.sample(range(len(self.circuits)), int(split*len(self.circuits)))
        val_circuits = [ self.__getitem__(i) for i in range(self.__len__()) if i in idx ]
        train_circuits = [ self.__getitem__(i) for i in range(self.__len__()) if i not in idx ]
        if self.mode == 'rep' or self.mode == 'noisy_rep':
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
        elif self.mode == 'noisy_rep':
            return self.noisy_circ_rep[i]
    
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


            
# DOESN'T WORK WITH CX AND CZ YET
class CircuitRepresentation(object):
    """Object for mapping qibo circuits to numpy array representation and vice versa."""
    def __init__(self, primitive_gates, noise_channels, shape='2d'):
        super(CircuitRepresentation, self).__init__()

        self.gate2index = { getattr(gates, g): i for i,g in enumerate(primitive_gates) }
        self.index2gate = { v: k for k,v in self.gate2index.items() }
        self.channel2index = {
            getattr(gates, c): i + len(self.gate2index) + 1
            for i,c in enumerate(noise_channels)
        }
        self.index2channel = { v: k for k,v in self.channel2index.items() }
        self.encoding_dim = len(primitive_gates) + 1 + len(noise_channels) + 1
        self.shape = shape

    # DOESN'T WORK WITH CX AND CZ
    def gate_to_array(self, gate):
        """Provide the one-hot encoding of a gate."""
        one_hot = np.zeros(self.encoding_dim)
        if type(gate) in self.channel2index.keys():
            gate_idx = self.channel2index[type(gate)]
            param_idx = self.encoding_dim - 1
            param_val = gate.init_kwargs['lam']
        elif type(gate) in self.gate2index.keys():
            gate_idx = self.gate2index[type(gate)]
            param_idx = len(self.gate2index)
            param_val = gate.init_kwargs['theta'] / (2 * np.pi)
        elif gate is None:
            return one_hot
        one_hot[gate_idx] = 1
        one_hot[param_idx] = param_val
        return one_hot

    # DOESN'T WORK WITH CX AND CZ
    def reorder_moments(self, moments):
        """Reorder circuit moments to group them in pairs (gate, channel)."""
        new_moments = []
        for row in list(zip(*moments)):
            new_row = []
            i = 0
            while i < len(row):
                if i == len(row) - 1:
                    if type(row[i]) in self.gate2index.keys():
                        new_row.append(row[i])
                        new_row.append(None)
                    break
                new_row.append(row[i])
                # check whether the next gate is a noise channel
                if type(row[i+1]) in self.channel2index.keys() or row[i+1] is None:
                    new_row.append(row[i+1])
                    i += 2
                # if it's not insert a None
                else:
                    new_row.append(None)
                    i += 1
            new_moments.append(new_row)
        return list(zip(*new_moments))

    def circuit_to_array(self, circuit):
        """Maps qibo circuits to numpy array representation.
        """
        nqubits = circuit.nqubits
        moments = circuit.queue.moments
        moments = self.reorder_moments(moments)
        rep = np.asarray([
            self.gate_to_array(gate)
            for m in list(zip(*moments))
            for gate in m
        ])
        rep = (rep[::2] + rep[1::2]).reshape(nqubits,-1,self.encoding_dim)
        rep = np.transpose(rep, axes=(1,0,2))
        if self.shape == '2d':
            rep = rep.reshape(-1, self.encoding_dim*nqubits)
        return rep

    # Doesn't work with CZ or CX
    def array_to_gate(self, array, qubit):
        """Build pair of qibo (gate,channel) starting from the encoding."""
        # separate gate part and channel part
        gate = array[:len(self.gate2index) + 1]
        channel = array[len(self.gate2index) + 1:]
        # extract parameters and objects
        theta = gate[-1]
        gate = self.index2gate[ int(gate[:-1].nonzero()[0]) ]
        gate = gate(qubit, theta=theta * 2*np.pi)
        # check whether there is a noisy channel
        if len(channel.nonzero()[0]) > 0:
            lam = channel[-1]
            channel = self.index2channel[
                int(channel[:-1].nonzero()[0]) + len(self.gate2index) + 1
            ]
            channel = channel([qubit], lam=lam)
        else:
            channel = None
        return (gate, channel)

    def array_to_circuit(self, array):
        """Maps numpy array to qibo circuit."""
        depth = array.shape[0]
        if self.shape == '2d':
            array = array.reshape(depth, -1, self.encoding_dim)
        nqubits = array.shape[1]
        c = Circuit(nqubits, density_matrix=True)
        for moment in array:
            # generate the gates and channels
            for qubit, row in enumerate(moment):
                gate, channel = self.array_to_gate(row, qubit)
                c.add(gate)
                if channel is not None:
                    c.add(channel)
        return c