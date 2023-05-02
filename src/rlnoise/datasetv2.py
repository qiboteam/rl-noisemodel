import itertools, json, random
import numpy as np
from qibo import gates
from qibo.models import Circuit
from inspect import signature
from rlnoise.rewards.classical_shadows import ClassicalShadows


class Dataset(object):

    def __init__(self, n_circuits, n_gates, n_qubits, representation, clifford=True, shadows=False, noise_model=None, mode='rep'):
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
        self.shadows = shadows
        
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
        '''
        self.noisy_circ_rep = np.asarray([
            self.rep.circuit_to_array(c)
            for c in self.noisy_circuits
        ],dtype=object)
        '''
        self.train_circuits, self.val_circuits ,self.train_noisy_label, self.val_noisy_label= self.train_val_split()
        

    def get_dm_labels(self, num_snapshots=10000):
        if self.shadows:
            states = []
            for i in range(self.n_circuits):
                model = ClassicalShadows(self.noisy_circuits[i], num_snapshots)
                model.get_classical_shadow()
                states.append(model.shadow_state_reconstruction())
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
                gate = random.choice(list(self.rep.index2gate.values()))
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
        #elif self.mode == 'noisy_rep': #TO BE FIXED TO WORK WITH MORE THAN ONE CHANNEL ON A GATE
        #    return self.noisy_circ_rep[i]
    
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
    def __init__(self, primitive_gates, noise_channels, shape='3d',coherent_noise=False):
        super(CircuitRepresentation, self).__init__()
        self.coherent_noise=coherent_noise
        self.gate2index = { getattr(gates, g): i for i,g in enumerate(primitive_gates) }
        self.index2gate = { v: k for k,v in self.gate2index.items() }
        self.channel2index = {
            getattr(gates, c): i + len(self.gate2index) + 1
            for i,c in enumerate(noise_channels)
        }
        self.index2channel = { v: k for k,v in self.channel2index.items() }
        self.encoding_dim = len(primitive_gates) + 1 + len(noise_channels) 
        if self.coherent_noise is True:
            self.encoding_dim+=2#(epsilon_Z,epsilon_X)
        self.shape = shape

    # DOESN'T WORK WITH CX AND CZ
    def gate_to_array(self, gate):
        
        """Provide the one-hot encoding of a gate."""
        one_hot = np.zeros(self.encoding_dim)
        if type(gate) in self.channel2index.keys():            
            #gate_idx = self.channel2index[type(gate)]
            #param_idx = self.encoding_dim - 1
            
            param_idx = self.channel2index[type(gate)]
            if type(gate) is gates.channels.DepolarizingChannel:               
                param_val = gate.init_kwargs['lam']
            if type(gate) is gates.channels.ThermalRelaxationChannel:
                param_val = gate.init_args[-1]#the last element is the time parameter
                
        elif type(gate) in self.gate2index.keys():
            gate_idx = self.gate2index[type(gate)]
            one_hot[gate_idx] = 1
            param_idx = len(self.gate2index)
            if 'theta' in gate.init_kwargs:
                param_val = gate.init_kwargs['theta'] / (2 * np.pi)
            else:
                param_val=0
        elif gate is None:
            return one_hot
        
        one_hot[param_idx] = param_val
        return one_hot

    #WORK WITH CX AND CZ

    def reorder_moments(self,moments):
        new_moments = []  
        count=0
        for row in list(zip(*moments)):
            
            count+=1
            row=np.array(row)
            new_row = []
            i = 0
            while i < len(row):
                if i == len(row) - 1:
                    if type(row[i]) in self.gate2index.keys() or row[i] is None:
                        new_row.append(row[i])
                        new_row.append(None)
                    if type(row[i]) in self.channel2index.keys():
                        new_row.append(None)
                        new_row.append(row[i])
                    break
                new_row.append(row[i])
                              # check whether the next gate is a noise channel
                if type(row[i+1]) in self.channel2index.keys():
                    new_row.append(row[i+1])
                    row[i+1]=None
                else:
                    new_row.append(None)
                i+=1
            new_moments.append(new_row)
        return list(zip(*new_moments))      
    #   DOESN'T WORK WITH MORE THAN ONE CHANNEL ON A GATE
    def circuit_to_array(self, circuit):
        """Maps qibo circuits to numpy array representation.
        """
        nqubits = circuit.nqubits
        moments = circuit.queue.moments
        moments2 = self.reorder_moments(moments)
        rep = np.asarray([
            self.gate_to_array(gate)
            for m in list(zip(*moments2))
            for gate in m
        ])     

        rep = (rep[::2] + rep[1::2]).reshape(nqubits,-1,self.encoding_dim)
        rep = np.transpose(rep, axes=(1,0,2))
        if self.shape == '2d':
            rep = rep.reshape(-1, self.encoding_dim*nqubits)
        clean_rep=np.asarray([rep[i] for i in range(len(rep)) if len(np.nonzero(rep[i])[0])!=0]) #added to exclude entries with all zeros
        return clean_rep #rep shape: n_moments,n_qubits,encoding_dim

    # work with CZ
    def array_to_gate(self, array, qubit,qubit2=None):
        """Build pair of qibo (gate,channel) starting from the encoding."""
        # separate gate part and channel part
        gate_arr = array[:len(self.gate2index) + 1]
        coherent_err_Z=None
        coherent_err_X=None
        if self.coherent_noise is True:
            channel_arr = array[len(self.gate2index) + 1:-2]
            epsilonZ=array[-2]#to be generalized
            epsilonX=array[-1]
        else:
            channel_arr = array[len(self.gate2index) + 1:]
            epsilonZ=None
            epsilonX=None
        # extract parameters and objects
        theta = gate_arr[-1]
        
        gate_idx=gate_arr[:-1].nonzero()[0]

        if (self.gate2index.get(gates.CZ) is not None and gate_arr[self.gate2index.get(gates.CZ)]==1) or (self.gate2index.get(gates.CNOT) is not None and gate_arr[self.gate2index.get(gates.CNOT)]==1): 
            gate = self.index2gate[ int(gate_idx) ]
            gate = gate(qubit,qubit2)
        elif gate_arr[self.gate2index.get(gates.RZ)]==1: 
            gate = self.index2gate[ int(gate_idx) ]
            gate = gate(qubit, theta=theta * 2*np.pi)    
           # if self.coherent_noise is True:
           #     gate_coher_err= gates.RZ(qubit, theta=array[-2] * 2*np.pi) 
           
        elif  gate_arr[self.gate2index.get(gates.RX)]==1:
            gate = self.index2gate[ int(gate_idx) ]
            gate = gate(qubit, theta=theta * 2*np.pi)
            #if self.coherent_noise is True:
            #    gate_coher_err= gates.RX(qubit, theta=array[-1] * 2*np.pi) 
         
        else:
            gate=None
        # check whether there is a noisy channel
        if len(channel_arr.nonzero()[0]) > 0:           
            channel_list=[]
            channel_idx=channel_arr.nonzero()[0]
            for idx in channel_idx:        
            #channel_idx=int(channel_idx)          
                channel = self.index2channel[
                    idx + len(self.gate2index) + 1
                ]          
                if channel is gates.channels.ThermalRelaxationChannel:
                    time=channel_arr[idx]
                    channel = channel(q=qubit,t1=1,t2=1, time=time)
                elif channel is gates.channels.DepolarizingChannel:
                    
                    lam=channel_arr[idx]
                    channel = channel([qubit], lam=lam)

                channel_list.append(channel)
        else:
            channel_list = None
        if epsilonZ is not None and epsilonZ!=0 :
            coherent_err_Z=gates.RZ(q=qubit,theta=epsilonZ)
        else: 
            coherent_err_Z=None
        if epsilonX is not None and epsilonX!=0 :
            coherent_err_X=gates.RX(q=qubit,theta=epsilonX)
        else:
            coherent_err_X=None
        gates_arr=[gate,coherent_err_Z,coherent_err_X]
        return (gates_arr, channel_list)


    def rep_to_circuit(self,rep_array):
        '''Maps numpy array to qibo circuit
        
        Args: 
            rep_array: array representation of the circuit
        Returns:
            c: qibo circuit corresponding to the array representation
        ''' 
        nqubits = rep_array.shape[1]      
        c = Circuit(nqubits, density_matrix=True)
        #print('rep array shape: ',rep_array.shape)
        for moment in range(len(rep_array)):
            count=-1
            pending_gates=[] #possible problem if using more than 3 qubits 
            for qubit, row in enumerate(rep_array[moment]):

                if self.gate2index.get(gates.CZ) is not None and row[int(self.gate2index.get(gates.CZ))]==1: 
                    if count == -1:
                        gate, channels=self.array_to_gate(row,qubit) 
                        for idx,i in enumerate(gate):
                            if i is not None and idx>0: #the first element is CZ gate and it will be added later, now is excluded
                                pending_gates.append(i)
                        count=qubit
                        if self.coherent_noise is False:
                            lam1=row[-2]
                            time1=row[-1]#to be generalized without adding the index manually
                        else:
                            lam1=row[-4]
                            time1=row[-3]#to be generalized without adding the index manually
                        pass
                    elif count!=-1:
                        gate, channels=self.array_to_gate(row,qubit,count) 
                        for i in range(len(gate)):
                            if gate[i] is not None:
                                c.add(gate[i]) #now gate is an array of form (TrueGate,coherent_err_Z,coherent_err_X)
                            if i<len(pending_gates) and pending_gates[i] is not None:
                                c.add(pending_gates[i])
                        if time1 !=0:
                            c.add(gates.ThermalRelaxationChannel(q=count,t1=1,t2=1,time=time1))
                        if channels is not None:
                            for channel in channels:
                                if channel.__class__ is gates.channels.DepolarizingChannel:
                                    c.add(channel.__class__((qubit,count),lam=(lam1+channel.init_kwargs['lam'])/2))#+channel.init_kwargs['lam'])/2)
                                elif channel.__class__ is gates.channels.ThermalRelaxationChannel:
                                    #print('added thermal relax')
                                    c.add(channel.__class__(q=qubit,t1=1,t2=1,time=channel.init_args[-1]))
        
                else:
                    gate_arr, channel = self.array_to_gate(row, qubit)
                    for gate in gate_arr:
                        if gate is not None:
                            c.add(gate)
                    if channel is not None:
                        c.add(channel)

                    
        return c

'''
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
'''

#IT WORKS ONLY WITH '3d' REPRESENTATION
#it seems to work with multi qubit and CZ and CX gates. 
#problem:
#0) reorder_moments() should be changed
#1) It add more gates than the specified depth beacuse the CZ or CX gate is counted only in one of the two connected qubit
#2) If we want to add more primitive gates we must modify line 284 and 311

#CHECK IF THE LABEL ARE CALCULATED CORRECTLY: generate the true label form qibo then generate the circ_rep convert in circ thanks to method in class and extraxt the label

