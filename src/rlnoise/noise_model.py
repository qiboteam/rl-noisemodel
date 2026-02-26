"""Noise model implementation for quantum circuits."""

from typing import List, Optional
from qibo import gates
from qibo.models import Circuit
from qibo.noise import NoiseModel, DepolarizingError, ResetError

from rlnoise.config import NoiseConfig


class QuantumNoiseModel:
    """Apply noise to quantum circuits based on configuration.
    
    This class provides a flexible way to add various types of noise
    to quantum circuits, including:
    - Depolarizing noise
    - Reset/damping errors
    - Coherent X and Z errors
    
    Args:
        config: NoiseConfig object specifying noise parameters
        qubits: Number of qubits in the system
    """
    
    def __init__(self, config: NoiseConfig, qubits: int):
        self.config = config
        self.qubits = qubits
        
    def _validate_config(self):
        """Validate that noise configuration is compatible with primitive gates.
        
        Note: This method is kept for backward compatibility but validation
        is now done in DatasetGenerator.
        \"\"\"
        pass
    
    @staticmethod
    def _string_to_gate(gate_string: str) -> Optional[type]:
        """Convert gate string to Qibo gate class.
        
        Args:
            gate_string: Name of the gate (case-insensitive)
            
        Returns:
            Qibo gate class or None
            
        Raises:
            ValueError: If gate name is not recognized
        """
        gate_map = {
            "none": None,
            "rx": gates.RX,
            "rz": gates.RZ,
            "cz": gates.CZ,
            "cnot": gates.CNOT,
        }
        
        gate_str_lower = gate_string.lower()
        if gate_str_lower not in gate_map:
            raise ValueError(f"Unrecognized gate: {gate_string}")
        
        return gate_map[gate_str_lower]
    
    def _apply_standard_noise(self, circuit: Circuit) -> Circuit:
        """Apply standard depolarizing and reset noise using Qibo's NoiseModel.
        
        Note: For per-qubit noise parameters (lists), the average value is used
        since Qibo's NoiseModel applies noise at the gate class level.
        Per-qubit coherent errors are applied individually in _add_coherent_errors().
        
        Args:
            circuit: Input quantum circuit
            
        Returns:
            Circuit with standard noise applied
        """
        noise_model = NoiseModel()
        noise_applied = False
        
        # Get noise parameters (average if list)
        damping_list = self.config.get_damping_list(self.qubits)
        depol_list = self.config.get_depolarizing_list(self.qubits)
        avg_damping = sum(damping_list) / len(damping_list)
        avg_depol = sum(depol_list) / len(depol_list)
        
        # Add reset/damping errors
        for gate_name in self.config.damping_on_gate:
            gate_class = self._string_to_gate(gate_name)
            if gate_class is not None:
                noise_model.add(ResetError(p0=avg_damping, p1=0), gate_class)
                noise_applied = True
        
        # Add depolarizing errors
        for gate_name in self.config.depol_on_gate:
            gate_class = self._string_to_gate(gate_name)
            if gate_class is not None:
                noise_model.add(DepolarizingError(avg_depol), gate_class)
                noise_applied = True
        
        return noise_model.apply(circuit) if noise_applied else circuit
    
    def _add_coherent_errors(self, circuit: Circuit, noisy_circuit: Circuit):
        """Add coherent X and Z rotation errors to gates.
        
        Args:
            circuit: Original circuit (unused but kept for clarity)
            noisy_circuit: Circuit to add coherent errors to (modified in place)
        """
        gates_to_process = list(noisy_circuit.queue)
        
        # Get noise parameters as lists
        coherent_x_list = self.config.get_coherent_x_list(self.qubits)
        coherent_y_list = self.config.get_coherent_y_list(self.qubits)
        
        for gate in gates_to_process:
            # Add coherent X errors
            if self.config.x_coherent_on_gate:
                for target_gate_name in self.config.x_coherent_on_gate:
                    if type(gate) == self._string_to_gate(target_gate_name):
                        if "theta" in gate.init_kwargs:
                            qubit = gate.qubits[0]
                            theta = coherent_x_list[qubit] * gate.init_kwargs["theta"]
                            noisy_circuit.add(gates.RX(qubit, theta=theta))
            
            # Add coherent Z errors
            if self.config.z_coherent_on_gate:
                for target_gate_name in self.config.z_coherent_on_gate:
                    if type(gate) == self._string_to_gate(target_gate_name):
                        if "theta" in gate.init_kwargs:
                            qubit = gate.qubits[0]
                            theta = coherent_y_list[qubit] * gate.init_kwargs["theta"]
                            noisy_circuit.add(gates.RZ(qubit, theta=theta))
    
    def apply(self, circuit: Circuit) -> Circuit:
        """Apply complete noise model to a quantum circuit.
        
        This method applies both standard noise (depolarizing, reset) and
        coherent errors according to the configuration.
        
        Args:
            circuit: Input quantum circuit
            
        Returns:
            New circuit with noise applied
        """
        # First apply standard noise using Qibo's NoiseModel
        intermediate_circuit = self._apply_standard_noise(circuit)
        
        # Create new circuit to add coherent errors
        noisy_circuit = Circuit(circuit.nqubits, density_matrix=True)
        
        # Copy all gates from intermediate circuit
        for gate in intermediate_circuit.queue:
            noisy_circuit.add(gate)
        
        # Add coherent errors
        self._add_coherent_errors(circuit, noisy_circuit)
        
        return noisy_circuit

