import itertools, json, random
import numpy as np
from qibo import gates
from qibo.models import Circuit
from inspect import signature
from rlnoise.rewards.classical_shadows import ClassicalShadows
from rlnoise.rewards.state_tomography import StateTomography
from configparser import ConfigParser

params=ConfigParser()
params.read("src/rlnoise/config.ini")

class Dataset(object):
    def __init__(self, n_circuits, n_gates, n_qubits, representation, clifford=True, shadows=False, readout_mit=False, noise_model=None, mode='rep', backend=None):
        '''Generate dataset for the training of RL-algorithm
        Args:
            n_circuits (int): number of random circuits generated
            n_gates (int): number of gates to add on each qubit
            n_qubits (int): number of qubits in each circuit
            representation: object of class CircuitRepresentation()
        '''
        super(Dataset, self).__init__()

        self.n_gates = n_gates
        self.n_qubits = n_qubits
        self.rep = representation
        self.clifford = clifford
        self.noise_model = noise_model
        self.mode = mode
        self.n_circuits=n_circuits
        self.shadows = shadows
        self.readout_mit = readout_mit
        self.backend = backend
        self.tomography=False
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
        ],dtype=object)
        self.train_circuits, self.val_circuits ,self.train_noisy_label, self.val_noisy_label= self.train_val_split()
        
    def get_dm_labels(self, num_snapshots=10000):
        if self.shadows:
            states = []
            for i in range(self.n_circuits):
                model = ClassicalShadows(self.noisy_circuits[i], num_snapshots)
                model.get_classical_shadow(backend=self.backend)
                states.append(model.shadow_state_reconstruction())
        elif self.tomography:
            states = []
            for i in range(self.n_circuits):
                model = StateTomography(nshots=num_snapshots, backend=self.backend)
                model.get_circuits(self.noisy_circuits[i])
                model.meas_obs(readout_mit=self.readout_mit)
                states.append(model.get_rho())
        else:
            states = [self.noisy_circuits[i]().state() for i in range(self.n_circuits)]
        return np.asarray(states)

    def get_frequencies(self, nshots=1000):
        assert self.mode == 'circ' or self.mode == 'noisy_circ'
        freq = []
        for i in range(self.__len__()):
            c = self.__getitem__(i)
            c.add(gates.M(*range(self.n_qubits)))
            freq.append(c(nshots=nshots).frequencies())
        return iter(freq)

    def generate_random_circuit(self):
        """Generate a random circuit."""
        circuit = Circuit(self.n_qubits, density_matrix=True)
        for _ in range(self.n_gates):
            for q0 in range(self.n_qubits):
                gate = random.choice([gates.RX, gates.RZ])#, gates.CZ])
                params = signature(gate).parameters
                # 2 qubit gate
                if 'q0' in params and 'q1' in params:
                    q1 = random.choice(
                        list( set(range(self.n_qubits)) - {q0} )
                    )                    
                    if 'theta' in params:
                        theta = random.choice([0, 0.25, 0.5, 0.75])
                        circuit.add(gate(q1,q0,theta))
                    else:
                        circuit.add(gate(q1,q0))
                else:
                    if issubclass(gate, gates.ParametrizedGate):
                        if self.clifford:
                            theta = random.choice([0, 0.25, 0.5, 0.75])
                        else:
                            theta = np.random.random()
                        theta *= 2 * np.pi
                        circuit.add(gate(q0, theta=theta))
                    else:
                        circuit.add(gate(q0))
        return circuit

    def train_val_split(self, split=0.2):
        '''
        Split dataset into train ad validation sets and it return the array representation of the circuits without noise
        and the label of the correspondent noisy circuit.

        Args:
            split: percentage of the total circuits that goes in validation set
        Return: 
            train_circuits: array representation of training circuit dataset
            val_circuits: same as before but for validation dataset
            train_noisy_label: labels of noisy circuits for training dataset
            val_noisy_label: same as before but for validation
        '''
        self.set_mode('rep')
        idx = random.sample(range(len(self.circuits)), int(split*len(self.circuits)))
        val_circuits = [ self.__getitem__(i) for i in range(self.__len__()) if i in idx ]
        train_circuits = [ self.__getitem__(i) for i in range(self.__len__()) if i not in idx ]
        val_label=[self.get_dm_labels()[i] for i in range(self.__len__()) if i in idx]
        train_label=[self.get_dm_labels()[i] for i in range(self.__len__()) if i not in idx]
        val_circuits = np.asarray(val_circuits,dtype=object)
        train_circuits = np.asarray(train_circuits,dtype=object)
        return train_circuits, val_circuits,train_label,val_label 
        
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

def gate_to_idx(gate):
        if gate is gates.RX:
            return 0
        if gate is gates.RZ:
            return 1
        if gate is gates.CZ:
            return 2
        if gate =="param":
            return 3
        if gate is gates.DepolarizingChannel:
            return 4
        if gate is gates.ResetChannel:
            return 5
        if gate =="epsilon_z":
            return 6
        if gate == "epsilon_x":
            return 7

class CircuitRepresentation(object):
    """
    Object for mapping qibo circuits to numpy array representation and vice versa.
    """  
    def __init__(self):
        self.encoding_dim = 8

    def gate_to_array(self, gate):
        """Provide the one-hot encoding of a gate."""
        one_hot = np.zeros(self.encoding_dim)
        if gate is None:
            return one_hot
        gate_idx = gate_to_idx(type(gate))
        one_hot[gate_idx] = 1
        param_idx = gate_to_idx("param")
        if 'theta' in gate.init_kwargs:
            one_hot[param_idx] = gate.init_kwargs['theta'] / (2 * np.pi)
        return one_hot  

    def circuit_to_array(self, circuit):
        """
        Maps non noisy qibo circuits to numpy array representation.
        """
        rep = np.asarray([
            self.gate_to_array(gate)
            for m in circuit.queue.moments
            for gate in m
        ])  
        rep = rep.reshape(-1, circuit.nqubits, self.encoding_dim)
        return rep

    def array_to_gate(self, array, qubit, qubit2=None):
        """Build pair of qibo (gate,channels) starting from the encoding.

        Args: 
            array: 1-D array that is the representation of a specific moment of the circuit
            qubit: the qubit that corresponds to that moment
        """
        channel_list=[]
        if array[gate_to_idx(gates.RX)] == 1:
            gate = gates.RX(qubit, theta=array[gate_to_idx('param')]*2*np.pi)
        elif array[gate_to_idx(gates.RZ)] == 1:
            gate = gates.RZ(qubit, theta=array[gate_to_idx('param')]*2*np.pi)
        elif array[gate_to_idx(gates.CZ)] == 1:
            gate = gates.CZ(qubit, qubit2)
        else:
            gate = None
        if array[gate_to_idx("epsilon_x")] != 0:
            channel_list.append(gates.RX(qubit, theta=array[gate_to_idx("epsilon_x")]))
        if array[gate_to_idx("epsilon_z")] != 0:
            channel_list.append(gates.RZ(qubit, theta=array[gate_to_idx("epsilon_z")]))
        if array[gate_to_idx(gates.ResetChannel)] != 0:
            channel_list.append(gates.ResetChannel(q=qubit, p0=array[gate_to_idx(gates.ResetChannel)], p1=0))
        if array[gate_to_idx(gates.DepolarizingChannel)] != 0:
            channel_list.append(gates.DepolarizingChannel([qubit], lam=array[gate_to_idx(gates.DepolarizingChannel)]))
        return (gate, channel_list)

    def rep_to_circuit(self,rep_array):
        '''Maps numpy array to qibo circuit
        
        Args: 
            rep_array: array representation of the circuit
        Returns:
            c: qibo circuit corresponding to the array representation
        '''     
        c = Circuit(rep_array.shape[1], density_matrix=True)
        for moment in range(len(rep_array)):
            count=-1
            for qubit, row in enumerate(rep_array[moment]):
                if row[gate_to_idx(gates.CZ)]==1: 
                    if count == -1:
                        _, pending_channels = self.array_to_gate(row,qubit)
                        count=qubit        
                        lam_0=row[gate_to_idx(gates.DepolarizingChannel)]                   
                    elif count != -1:
                        gate, channels = self.array_to_gate(row, qubit, count) 
                        lam_1=row[gate_to_idx(gates.DepolarizingChannel)]
                        c.add(gate)
                        for channel in pending_channels[0:-1]:
                            c.add(channel)
                        for channel in channels[0:-1]:
                            c.add(channel)
                        avg_lam=(lam_0+lam_1)/2.
                        if avg_lam !=0:
                            c.add(gates.DepolarizingChannel((count, qubit), lam=avg_lam))
                else:
                    gate, channels = self.array_to_gate(row, qubit)
                    if gate is not None:
                        c.add(gate)
                    for channel in channels:
                        c.add(channel)
        return c