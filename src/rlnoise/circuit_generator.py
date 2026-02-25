"""Quantum circuit generation utilities."""

import random
from typing import List
import numpy as np
from qibo import gates
from qibo.models import Circuit
from qibo.quantum_info.random_ensembles import random_clifford

from rlnoise.config import DatasetConfig


class CircuitGenerator:
    """Generate quantum circuits for dataset creation.
    
    This class provides methods to generate various types of quantum circuits:
    - Random circuits with arbitrary angles
    - Clifford circuits (quantized angles)
    - Circuits decomposed into primitive gates
    
    Args:
        config: DatasetConfig object specifying circuit parameters
    """
    
    def __init__(self, config: DatasetConfig, primitive_gates: List[str]):
        self.config = config
        self.primitive_gates = [gate.lower() for gate in primitive_gates]
        self.n_qubits = config.qubits
        self.n_moments = config.moments
        self.is_clifford = config.clifford
        
        # Validate configuration
        if self.n_qubits < 2 and "cz" in self.primitive_gates:
            raise ValueError("Cannot use CZ gates on single-qubit circuits")
    
    def generate_random_circuit(self) -> Circuit:
        """Generate a random quantum circuit.
        
        Creates a circuit with random gates from the primitive gate set.
        For Clifford circuits, angles are quantized to [0, π/2, π, 3π/2].
        For non-Clifford circuits, angles are uniformly random in [0, 2π].
        
        Returns:
            Random quantum circuit
        """
        circuit = Circuit(self.n_qubits, density_matrix=True)
        
        while len(circuit.queue.moments) < self.n_moments:
            gate = self._generate_random_gate()
            if gate is not None:
                circuit.add(gate)
        
        return circuit
    
    def _generate_random_gate(self):
        """Generate a single random gate from the primitive gate set.
        
        Returns:
            Random gate instance or None
        """
        qubit = random.choice(range(self.n_qubits))
        gate_type = random.choice(self.primitive_gates)
        
        if gate_type == "cz":
            # Select a different qubit for two-qubit gate
            other_qubit = random.choice([q for q in range(self.n_qubits) if q != qubit])
            return gates.CZ(qubit, other_qubit)
        
        elif gate_type == "cnot":
            other_qubit = random.choice([q for q in range(self.n_qubits) if q != qubit])
            return gates.CNOT(qubit, other_qubit)
        
        elif gate_type == "rx":
            theta = self._sample_angle()
            return gates.RX(qubit, theta=theta)
        
        elif gate_type == "rz":
            theta = self._sample_angle()
            return gates.RZ(qubit, theta=theta)
        
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")
    
    def _sample_angle(self) -> float:
        """Sample rotation angle based on circuit type.
        
        Returns:
            Rotation angle in radians
        """
        if self.is_clifford:
            # Clifford: use quantized angles
            return random.choice([0, 0.25, 0.5, 0.75]) * 2 * np.pi
        else:
            # Non-Clifford: uniform random angle
            return np.random.random() * 2 * np.pi
    
    def generate_clifford_circuit(self) -> Circuit:
        """Generate a random Clifford circuit using Qibo's built-in generator.
        
        The circuit is decomposed into primitive gates specified in the configuration.
        
        Returns:
            Random Clifford circuit decomposed into primitive gates
        """
        # Generate random Clifford circuit using Qibo
        raw_circuit = random_clifford(self.n_qubits, return_circuit=True, density_matrix=True)
        
        # Decompose into primitive gates
        decomposed_circuit = Circuit(self.n_qubits, density_matrix=True)
        
        for gate in raw_circuit.queue:
            self._decompose_gate(gate, decomposed_circuit)
        
        return decomposed_circuit
    
    def _decompose_gate(self, gate, target_circuit: Circuit):
        """Decompose a gate into primitive gates.
        
        Args:
            gate: Gate to decompose
            target_circuit: Circuit to add decomposed gates to
        """
        gate_name = gate.name.lower()
        
        if gate_name in self.primitive_gates:
            # Gate is already primitive
            target_circuit.add(gate)
        
        elif gate_name == "cx":
            # CNOT decomposition: RZ(π/2) - RX(π/2) - CZ - RZ(π/2) - RX(π/2)
            control, target = gate.qubits[0], gate.qubits[1]
            target_circuit.add(gates.RZ(target, np.pi / 2))
            target_circuit.add(gates.RX(target, np.pi / 2))
            target_circuit.add(gates.CZ(control, target))
            target_circuit.add(gates.RZ(target, np.pi / 2))
            target_circuit.add(gates.RX(target, np.pi / 2))
        
        elif gate_name == "h":
            # Hadamard decomposition: RZ(π/2) - RX(π/2)
            qubit = gate.qubits[0]
            target_circuit.add(gates.RZ(qubit, np.pi / 2))
            target_circuit.add(gates.RX(qubit, np.pi / 2))
        
        elif gate_name == "z":
            # Z gate: RZ(π)
            qubit = gate.qubits[0]
            target_circuit.add(gates.RZ(qubit, np.pi))
        
        elif gate_name == "y":
            # Y gate: RZ(π) - RX(π)
            qubit = gate.qubits[0]
            target_circuit.add(gates.RZ(qubit, np.pi))
            target_circuit.add(gates.RX(qubit, np.pi))
        
        elif gate_name == "x":
            # X gate: RX(π)
            qubit = gate.qubits[0]
            target_circuit.add(gates.RX(qubit, np.pi))
        
        elif gate_name == "s":
            # S gate: RZ(π/2)
            qubit = gate.qubits[0]
            target_circuit.add(gates.RZ(qubit, np.pi / 2))
        
        else:
            raise ValueError(f"Cannot decompose unknown gate: {gate_name}")
    
    def generate_batch(self, n_circuits: int, mixed: bool = False) -> List[Circuit]:
        """Generate a batch of circuits.
        
        Args:
            n_circuits: Number of circuits to generate
            mixed: If True, generate half random and half Clifford circuits
            
        Returns:
            List of generated circuits
        """
        if mixed:
            # Generate mixed dataset
            n_random = n_circuits // 2
            n_clifford = n_circuits - n_random
            
            circuits = (
                [self.generate_random_circuit() for _ in range(n_random)]
                + [self.generate_clifford_circuit() for _ in range(n_clifford)]
            )
            random.shuffle(circuits)
        
        elif self.config.distributed_clifford:
            # Generate only Clifford circuits
            circuits = [self.generate_clifford_circuit() for _ in range(n_circuits)]
        
        else:
            # Generate only random circuits
            circuits = [self.generate_random_circuit() for _ in range(n_circuits)]
        
        return circuits
