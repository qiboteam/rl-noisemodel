"""Circuit representation and encoding for machine learning."""

from typing import Optional, Tuple, List
import numpy as np
from qibo import gates
from qibo.models import Circuit


class CircuitEncoder:
    """Encode quantum circuits as numpy arrays for ML models.
    
    This class provides methods to convert Qibo circuits to and from
    numpy array representations suitable for neural network processing.
    
    The encoding uses one-hot vectors with the following dimensions:
    - Gate type (RZ, RX, CZ, etc.)
    - Gate parameter (for parametric gates)
    - Noise channels (depolarizing, reset, coherent errors)
    
    Args:
        primitive_gates: List of primitive gate names
        encoding_dim: Dimension of the encoding vector (default: 8)
    """
    
    # Gate type indices
    IDX_RZ = 0
    IDX_RX = 1
    IDX_CZ = 2
    IDX_PARAM = 3
    IDX_DEPOL = 4
    IDX_RESET = 5
    IDX_EPSILON_Z = 6
    IDX_EPSILON_X = 7
    
    def __init__(self, primitive_gates: List[str], encoding_dim: int = 8):
        self.primitive_gates = [g.lower() for g in primitive_gates]
        self.encoding_dim = encoding_dim
    
    def _gate_to_idx(self, gate) -> int:
        """Map gate type to index in encoding vector.
        
        Args:
            gate: Gate class, instance, or string identifier
            
        Returns:
            Index in the encoding vector
            
        Raises:
            ValueError: If gate type is unknown
        """
        gate_map = {
            gates.RZ: self.IDX_RZ,
            gates.RX: self.IDX_RX,
            gates.CZ: self.IDX_CZ,
            gates.DepolarizingChannel: self.IDX_DEPOL,
            gates.ResetChannel: self.IDX_RESET,
            "param": self.IDX_PARAM,
            "epsilon_z": self.IDX_EPSILON_Z,
            "epsilon_x": self.IDX_EPSILON_X,
        }
        
        if gate in gate_map:
            return gate_map[gate]
        
        raise ValueError(f"Unknown gate type: {gate}")
    
    def gate_to_array(self, gate, qubit: int) -> np.ndarray:
        """Convert a single gate to its array representation.
        
        Args:
            gate: Qibo gate instance (or None for identity)
            qubit: Qubit index the gate acts on
            
        Returns:
            One-hot encoded array of shape (encoding_dim,)
        """
        encoding = np.zeros(self.encoding_dim)
        
        if gate is None:
            return encoding
        
        gate_type = type(gate)
        gate_idx = self._gate_to_idx(gate_type)
        
        # Handle two-qubit gates (CZ, CNOT)
        if gate_type in [gates.CZ, gates.CNOT]:
            target_qubits = gate.target_qubits
            if qubit == target_qubits[0]:
                encoding[gate_idx] = -1  # Target qubit
            else:
                encoding[gate_idx] = 1  # Control qubit
        else:
            encoding[gate_idx] = 1
        
        # Encode gate parameter if present
        if "theta" in gate.init_kwargs:
            param_idx = self._gate_to_idx("param")
            encoding[param_idx] = gate.init_kwargs["theta"] / (2 * np.pi)
        
        return encoding
    
    def circuit_to_array(self, circuit: Circuit) -> np.ndarray:
        """Convert a quantum circuit to array representation.
        
        Args:
            circuit: Qibo circuit (without noise)
            
        Returns:
            Array of shape (n_moments, n_qubits, encoding_dim)
        """
        representations = []
        
        for moment in circuit.queue.moments:
            moment_rep = np.array([
                self.gate_to_array(gate, qubit)
                for qubit, gate in enumerate(moment)
            ])
            representations.append(moment_rep)
        
        # Stack to shape (n_moments, n_qubits, encoding_dim)
        return np.array(representations)
    
    def array_to_gate(
        self, array: np.ndarray, qubit: int, qubit2: Optional[int] = None
    ) -> Tuple[Optional[gates.Gate], List[gates.Gate]]:
        """Convert array encoding back to gate and noise channels.
        
        Args:
            array: Array encoding of shape (encoding_dim,)
            qubit: Primary qubit index
            qubit2: Secondary qubit index (for two-qubit gates)
            
        Returns:
            Tuple of (gate, channel_list) where:
                - gate: The quantum gate or None
                - channel_list: List of noise channels to apply
        """
        gate = None
        channels = []
        
        # Decode gate type
        if array[self.IDX_RX] == 1:
            theta = array[self.IDX_PARAM] * 2 * np.pi
            gate = gates.RX(qubit, theta=theta, trainable=False)
        
        elif array[self.IDX_RZ] == 1:
            theta = array[self.IDX_PARAM] * 2 * np.pi
            gate = gates.RZ(qubit, theta=theta, trainable=False)
        
        elif array[self.IDX_CZ] != 0 and qubit2 is not None:
            gate = gates.CZ(qubit, qubit2)
        
        # Decode noise channels
        if array[self.IDX_EPSILON_X] != 0:
            channels.append(gates.RX(qubit, theta=array[self.IDX_EPSILON_X], trainable=False))
        
        if array[self.IDX_EPSILON_Z] != 0:
            channels.append(gates.RZ(qubit, theta=array[self.IDX_EPSILON_Z], trainable=False))
        
        if array[self.IDX_RESET] != 0:
            channels.append(gates.ResetChannel(qubit, [array[self.IDX_RESET], 0]))
        
        if array[self.IDX_DEPOL] != 0:
            channels.append(gates.DepolarizingChannel([qubit], lam=array[self.IDX_DEPOL]))
        
        return gate, channels
    
    def array_to_circuit(self, array: np.ndarray) -> Circuit:
        """Convert array representation back to quantum circuit.
        
        Args:
            array: Array of shape (n_moments, n_qubits, encoding_dim)
            
        Returns:
            Qibo circuit with gates and noise channels
        """
        n_moments, n_qubits, _ = array.shape
        circuit = Circuit(n_qubits, density_matrix=True)
        
        for moment_idx in range(n_moments):
            moment = array[moment_idx]
            
            # Handle two-qubit gates
            cz_qubits = []
            for qubit_idx in range(n_qubits):
                if moment[qubit_idx, self.IDX_CZ] != 0:
                    cz_qubits.append(qubit_idx)
            
            if len(cz_qubits) == 2:
                # Process two-qubit gate
                self._add_two_qubit_gate(circuit, moment, cz_qubits)
            else:
                # Process single-qubit gates
                for qubit_idx in range(n_qubits):
                    self._add_single_qubit_gate(circuit, moment[qubit_idx], qubit_idx)
        
        return circuit
    
    def _add_two_qubit_gate(self, circuit: Circuit, moment: np.ndarray, qubits: List[int]):
        """Add a two-qubit gate and associated noise to the circuit.
        
        Args:
            circuit: Circuit to add gates to
            moment: Moment array of shape (n_qubits, encoding_dim)
            qubits: List of two qubit indices
        """
        q0, q1 = qubits
        
        # Determine control and target
        if moment[q0, self.IDX_CZ] == -1:
            target, control = q0, q1
        else:
            control, target = q0, q1
        
        # Get gates and channels for both qubits
        gate0, channels0 = self.array_to_gate(moment[q0], q0, q1)
        gate1, channels1 = self.array_to_gate(moment[q1], q1, q0)
        
        # Add the two-qubit gate
        if gate0 is not None:
            circuit.add(gate0)
        elif gate1 is not None:
            circuit.add(gate1)
        
        # Add single-qubit noise channels
        for channel in channels0[:-1]:  # Exclude depolarizing for now
            circuit.add(channel)
        for channel in channels1[:-1]:
            circuit.add(channel)
        
        # Add averaged two-qubit depolarizing noise
        lam0 = moment[q0, self.IDX_DEPOL]
        lam1 = moment[q1, self.IDX_DEPOL]
        avg_lam = (lam0 + lam1) / 2.0
        
        if avg_lam != 0:
            circuit.add(gates.DepolarizingChannel((control, target), lam=avg_lam))
    
    def _add_single_qubit_gate(self, circuit: Circuit, encoding: np.ndarray, qubit: int):
        """Add a single-qubit gate and noise to the circuit.
        
        Args:
            circuit: Circuit to add gates to
            encoding: Encoding array for this qubit
            qubit: Qubit index
        """
        gate, channels = self.array_to_gate(encoding, qubit)
        
        if gate is not None:
            circuit.add(gate)
        
        for channel in channels:
            circuit.add(channel)
