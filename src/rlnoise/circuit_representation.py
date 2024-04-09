from qibo import gates, Circuit
import json
import numpy as np

def gate_to_idx(gate):
    """Map gate to index in the circuit representation."""
    if gate is gates.RZ:
        return 0
    elif gate is gates.RX:
        return 1
    elif gate is gates.CZ:
        return 2
    elif gate == "param":
        return 3
    elif gate is gates.DepolarizingChannel:
        return 4
    elif gate is gates.ResetChannel:
        return 5
    elif gate == "epsilon_z":
        return 6
    elif gate == "epsilon_x":
        return 7
    else:
        raise ValueError(f"Unknown gate {gate}")

def gate_action_index(gate):
    """Map gate to index in the action representation."""
    if gate == 'epsilon_x':
        return 0
    elif gate == 'epsilon_z':
        return 1
    elif gate == gates.ResetChannel:
        return 2
    elif gate == gates.DepolarizingChannel:
        return 3  
    else:
        raise ValueError(f"Unknown gate {gate}")

class CircuitRepresentation(object):
    """
    Object for mapping qibo circuits to numpy array representation and vice versa.
    """  
    def __init__(self, config_file):
        with open(config_file) as f:
            self.config = json.load(f)
        self.encoding_dim = 8
        self.primitive_gates = self.config['noise']['primitive_gates']
        self.max_action = self.config["gym_env"]["action_space_max_value"]

    def gate_to_array(self, gate, qubit):
        """Provide the one-hot encoding of a gate."""
        one_hot = np.zeros(self.encoding_dim)
        if gate is None:
            return one_hot
        gate_idx = gate_to_idx(type(gate))
        if type(gate) is gates.CNOT and qubit == gate.target_qubits[0]:
            one_hot[gate_idx] = -1
        elif type(gate) is gates.CNOT:

            one_hot[gate_idx] = 1
        else:
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
            self.gate_to_array(gate,qubit)
            for m in circuit.queue.moments
            for qubit, gate in enumerate(m)
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
            channel_list.append(gates.ResetChannel(qubit, [array[gate_to_idx(gates.ResetChannel)], 0]))
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
            target = None
            for qubit, row in enumerate(rep_array[moment]):
                if row[gate_to_idx(gates.CZ)]==1: 
                    if count == -1:
                        _, pending_channels = self.array_to_gate(row,qubit)
                        count=qubit        
                        lam_0=row[gate_to_idx(gates.DepolarizingChannel)]
                    else:
                        gate, channels = self.array_to_gate(row, qubit, count)
                        lam_1=row[gate_to_idx(gates.DepolarizingChannel)]
                        c.add(gate)
                        for channel in pending_channels[:-1]:
                            c.add(channel)
                        for channel in channels[:-1]:
                            c.add(channel)
                        avg_lam=(lam_0+lam_1)/2.
                        if avg_lam !=0:
                            c.add(gates.DepolarizingChannel((count, qubit), lam=avg_lam))
                elif "CNOT" in self.primitive_gates and row[gate_to_idx(gates.CZ)] == -1 or row[gate_to_idx(gates.CZ)] == 1:
                    if row[gate_to_idx(gates.CZ)] == -1:
                        target = qubit
                    elif row[gate_to_idx(gates.CZ)] == 1:
                        control = qubit

                    if count == -1:
                        _, pending_channels = self.array_to_gate(row,qubit)
                        count=qubit        
                        lam_0=row[gate_to_idx(gates.DepolarizingChannel)]
                    else:
                        gate, channels = self.array_to_gate(row, control, target)
                        lam_1=row[gate_to_idx(gates.DepolarizingChannel)]
                        c.add(gate)
                        for channel in pending_channels[:-1]:
                            c.add(channel)
                        for channel in channels[:-1]:
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

    def make_action(self, action, circuit, position):
        """Apply the action to the circuit at a specific position."""
        nqubits = circuit.shape[1]
        for q in range(nqubits):          
            for idx, a in enumerate(action[q]):
                a *= self.max_action
                if idx == gate_action_index(gates.DepolarizingChannel):
                    circuit[gate_to_idx(gates.DepolarizingChannel), q, position] = a        
                if idx == gate_action_index("epsilon_x"):
                    circuit[gate_to_idx("epsilon_x"), q, position] = a
                if idx == gate_action_index("epsilon_z"):
                    circuit[gate_to_idx("epsilon_z"), q, position] = a
                if idx == gate_action_index(gates.ResetChannel):
                    circuit[gate_to_idx(gates.ResetChannel), q, position] = a        
        return circuit
