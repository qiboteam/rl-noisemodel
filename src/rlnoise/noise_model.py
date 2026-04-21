"""Noise model implementation for quantum circuits."""

from typing import Optional
from qibo import gates
from qibo.models import Circuit
from qibo.noise import NoiseModel, DepolarizingError, ResetError

from rlnoise.config import NoiseConfig


class QuantumNoiseModel:  # pylint: disable=too-few-public-methods
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
        self._validate_noise_parameters()

    def _validate_noise_parameters(self):
        """Validate that list-based noise parameters match the number of qubits.

        Raises:
            ValueError: If a noise parameter is a list with length != qubits
        """
        for noise_spec in self.config.noise_list:
            param = noise_spec.noise_parameter
            if isinstance(param, list):
                if len(param) != self.qubits:
                    raise ValueError(
                        f"Noise parameter list for {noise_spec.gate} gate "
                        f"({noise_spec.noise_channel}) has length {len(param)}, "
                        f"but circuit has {self.qubits} qubits. "
                        f"List length must match number of qubits."
                    )


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
        Per-qubit noise parameters are properly handled in _add_gate_specific_noise().

        Args:
            circuit: Input quantum circuit

        Returns:
            Circuit with standard noise applied
        """
        noise_model = NoiseModel()
        noise_applied = False

        # Group noise by gate and channel type for standard noise (damping/depolarizing)
        # We need to average per-qubit parameters for Qibo's gate-class-level noise
        damping_noise = self.config.get_noise_by_channel("damping")
        depol_noise = self.config.get_noise_by_channel("depolarizing")

        # Add reset/damping errors
        for noise in damping_noise:
            gate_class = self._string_to_gate(noise.gate)
            if gate_class is not None:
                # Get parameter as list and average it
                param_list = noise.get_parameter_list(self.qubits)
                avg_param = sum(param_list) / len(param_list)
                noise_model.add(ResetError(p0=avg_param, p1=0), gate_class)
                noise_applied = True

        # Add depolarizing errors
        for noise in depol_noise:
            gate_class = self._string_to_gate(noise.gate)
            if gate_class is not None:
                # Get parameter as list and average it
                param_list = noise.get_parameter_list(self.qubits)
                avg_param = sum(param_list) / len(param_list)
                noise_model.add(DepolarizingError(avg_param), gate_class)
                noise_applied = True

        return noise_model.apply(circuit) if noise_applied else circuit

    def _add_coherent_errors_for_gate(self, gate, noisy_circuit: Circuit):
        """Add coherent X and Z rotation errors for a specific gate.

        Coherent errors are added as additional RX or RZ gates immediately after the target gate.
        If angle_dependent is True, the error is scaled by the gate's rotation angle.
        Noise gates are marked with trainable=True to distinguish them from original gates.

        Args:
            gate: The gate to potentially add coherent errors after
            noisy_circuit: Circuit to add coherent errors to (modified in place)
        """
        # Get coherent noise configurations
        coherent_x_noise = self.config.get_noise_by_channel("coherent_x")
        coherent_z_noise = self.config.get_noise_by_channel("coherent_z")

        # Add coherent X errors
        for noise in coherent_x_noise:
            gate_class = self._string_to_gate(noise.gate)
            if gate_class is not None and isinstance(gate, gate_class):
                qubit = gate.qubits[0]
                param_list = noise.get_parameter_list(self.qubits)

                if noise.angle_dependent and "theta" in gate.init_kwargs:
                    # Scale error by gate angle
                    theta = param_list[qubit] * gate.init_kwargs["theta"]
                else:
                    # Fixed error magnitude
                    theta = param_list[qubit]

                # Mark as trainable=True to identify as noise gate
                noisy_circuit.add(gates.RX(qubit, theta=theta, trainable=True))

        # Add coherent Z errors
        for noise in coherent_z_noise:
            gate_class = self._string_to_gate(noise.gate)
            if gate_class is not None and isinstance(gate, gate_class):
                qubit = gate.qubits[0]
                param_list = noise.get_parameter_list(self.qubits)

                if noise.angle_dependent and "theta" in gate.init_kwargs:
                    # Scale error by gate angle
                    theta = param_list[qubit] * gate.init_kwargs["theta"]
                else:
                    # Fixed error magnitude
                    theta = param_list[qubit]

                # Mark as trainable=True to identify as noise gate
                noisy_circuit.add(gates.RZ(qubit, theta=theta, trainable=True))

    def apply(self, circuit: Circuit) -> Circuit:
        """Apply complete noise model to a quantum circuit.

        This method applies both standard noise (depolarizing, reset) and
        coherent errors according to the configuration.

        Coherent errors are added immediately after each gate to ensure proper
        gate ordering. Noise gates are marked with trainable=True to distinguish
        them from original gates (trainable=False).

        Args:
            circuit: Input quantum circuit

        Returns:
            New circuit with noise applied
        """
        # First apply standard noise using Qibo's NoiseModel
        intermediate_circuit = self._apply_standard_noise(circuit)

        # Create new circuit to add coherent errors
        noisy_circuit = Circuit(circuit.nqubits, density_matrix=True)

        # Copy gates and add coherent errors immediately after each gate
        for gate in intermediate_circuit.queue:
            # Add the original gate (or gate with standard noise channel)
            noisy_circuit.add(gate)

            # Immediately add coherent errors for this gate
            self._add_coherent_errors_for_gate(gate, noisy_circuit)

        return noisy_circuit
