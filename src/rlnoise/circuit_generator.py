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

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.primitive_gates = config.primitive_gates
        self.n_qubits = config.qubits
        self.n_moments = config.moments
        self.is_clifford = config.clifford

        # Validate configuration
        if self.n_qubits < 2 and "cz" in self.primitive_gates:
            raise ValueError(  # pragma: no cover
                "Cannot use CZ gates on single-qubit circuits"
            )  # pragma: no cover

    def generate_random_circuit(self) -> Circuit:
        """Generate a random quantum circuit.

        Creates a circuit with random gates from the primitive gate set.
        For Clifford circuits, angles are quantized to [0, Ï€/2, Ï€, 3Ï€/2].
        For non-Clifford circuits, angles are uniformly random in [0, 2Ï€].

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

        if gate_type == "cnot":
            other_qubit = random.choice([q for q in range(self.n_qubits) if q != qubit])
            return gates.CNOT(qubit, other_qubit)

        if gate_type == "rx":
            theta = self._sample_angle()
            return gates.RX(qubit, theta=theta, trainable=False)

        if gate_type == "rz":
            theta = self._sample_angle()
            return gates.RZ(qubit, theta=theta, trainable=False)

        raise ValueError(f"Unknown gate type: {gate_type}")

    def _sample_angle(self) -> float:
        """Sample rotation angle based on circuit type.

        Returns:
            Rotation angle in radians
        """
        if self.is_clifford:
            # Clifford: use quantized angles
            return random.choice([0.25, 0.5, 0.75]) * 2 * np.pi
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

    def _decompose_gate(self, gate, target_circuit: Circuit):  # pylint: disable=too-many-branches,too-many-statements
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
            # CNOT decomposition: RZ(Ï€/2) - RX(Ï€/2) - CZ - RZ(Ï€/2) - RX(Ï€/2)
            control, target = gate.qubits[0], gate.qubits[1]
            target_circuit.add(gates.RZ(target, np.pi / 2, trainable=False))
            target_circuit.add(gates.RX(target, np.pi / 2, trainable=False))
            target_circuit.add(gates.CZ(control, target))
            target_circuit.add(gates.RZ(target, np.pi / 2, trainable=False))
            target_circuit.add(gates.RX(target, np.pi / 2, trainable=False))

        elif gate_name == "h":
            # Hadamard decomposition: RZ(Ï€/2) - RX(Ï€/2)
            qubit = gate.qubits[0]
            target_circuit.add(gates.RZ(qubit, np.pi / 2, trainable=False))
            target_circuit.add(gates.RX(qubit, np.pi / 2, trainable=False))

        elif gate_name == "z":
            # Z gate: RZ(Ï€)
            qubit = gate.qubits[0]
            target_circuit.add(gates.RZ(qubit, np.pi, trainable=False))

        elif gate_name == "y":
            # Y gate: RZ(Ï€) - RX(Ï€)
            qubit = gate.qubits[0]
            target_circuit.add(gates.RZ(qubit, np.pi, trainable=False))
            target_circuit.add(gates.RX(qubit, np.pi, trainable=False))

        elif gate_name == "x":
            # X gate: RX(Ï€)
            qubit = gate.qubits[0]
            target_circuit.add(gates.RX(qubit, np.pi, trainable=False))

        elif gate_name == "s":
            # S gate: RZ(Ï€/2)
            qubit = gate.qubits[0]
            target_circuit.add(gates.RZ(qubit, np.pi / 2, trainable=False))

        elif gate_name == "sdg":
            # Sâ€  gate: RZ(-Ï€/2)
            qubit = gate.qubits[0]
            target_circuit.add(gates.RZ(qubit, -np.pi / 2, trainable=False))

        elif gate_name == "t":
            # T gate: RZ(Ï€/4)
            qubit = gate.qubits[0]
            target_circuit.add(gates.RZ(qubit, np.pi / 4, trainable=False))

        elif gate_name == "tdg":
            # Tâ€  gate: RZ(-Ï€/4)
            qubit = gate.qubits[0]
            target_circuit.add(gates.RZ(qubit, -np.pi / 4, trainable=False))

        elif gate_name == "sx":
            # âˆšX gate: RX(Ï€/2)
            qubit = gate.qubits[0]
            target_circuit.add(gates.RX(qubit, np.pi / 2, trainable=False))

        elif gate_name == "sxdg":
            # âˆšXâ€  gate: RX(-Ï€/2)
            qubit = gate.qubits[0]
            target_circuit.add(gates.RX(qubit, -np.pi / 2, trainable=False))

        elif gate_name == "swap":
            # SWAP decomposition via three CZ + Hadamards
            q0, q1 = gate.qubits[0], gate.qubits[1]
            # SWAP = CX(q0,q1) Â· CX(q1,q0) Â· CX(q0,q1)
            for ctrl, tgt in [(q0, q1), (q1, q0), (q0, q1)]:
                target_circuit.add(gates.RZ(tgt, np.pi / 2, trainable=False))
                target_circuit.add(gates.RX(tgt, np.pi / 2, trainable=False))
                target_circuit.add(gates.CZ(ctrl, tgt))
                target_circuit.add(gates.RZ(tgt, np.pi / 2, trainable=False))
                target_circuit.add(gates.RX(tgt, np.pi / 2, trainable=False))

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


# ---------------------------------------------------------------------------
# Fixed benchmark circuits (3 qubits, RX / RZ / CZ primitive basis)
# ---------------------------------------------------------------------------

def grover_circuit() -> Circuit:
    """Return a 3-qubit Grover search circuit targeting the |11x> subspace.

    The circuit is expressed in the RX / RZ / CZ primitive gate basis so that
    it can be processed by :class:`CircuitEncoder` and fed into the RL agent.
    The third qubit is an ancilla.

    Returns:
        Qibo :class:`Circuit` with ``density_matrix=True``.
    """
    circ = Circuit(3, density_matrix=True)
    # Hadamard on qubits 0 and 1  (H = RZ(π/2)·RX(π/2))
    circ.add(gates.RZ(0, np.pi / 2, trainable=False))
    circ.add(gates.RX(0, np.pi / 2, trainable=False))
    circ.add(gates.RZ(0, np.pi / 2, trainable=False))
    circ.add(gates.RZ(1, np.pi / 2, trainable=False))
    circ.add(gates.RX(1, np.pi / 2, trainable=False))
    circ.add(gates.RZ(1, np.pi / 2, trainable=False))
    # Initialise ancilla to |-> via X then H
    circ.add(gates.RX(2, np.pi, trainable=False))
    circ.add(gates.RZ(2, np.pi / 2, trainable=False))
    circ.add(gates.RX(2, np.pi / 2, trainable=False))
    circ.add(gates.RZ(2, np.pi / 2, trainable=False))
    # Toffoli (CCX) decomposition via CZ + RX rotations
    circ.add(gates.CZ(1, 2))
    circ.add(gates.RX(2, -np.pi / 4, trainable=False))
    circ.add(gates.CZ(0, 2))
    circ.add(gates.RX(2, np.pi / 4, trainable=False))
    circ.add(gates.CZ(1, 2))
    circ.add(gates.RX(2, -np.pi / 4, trainable=False))
    circ.add(gates.CZ(0, 2))
    circ.add(gates.RX(2, np.pi / 4, trainable=False))
    circ.add(gates.RZ(1, np.pi / 4, trainable=False))
    circ.add(gates.RZ(1, np.pi / 2, trainable=False))
    circ.add(gates.RX(1, np.pi / 2, trainable=False))
    circ.add(gates.RZ(1, np.pi / 2, trainable=False))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.RZ(0, np.pi / 4, trainable=False))
    circ.add(gates.RX(1, -np.pi / 4, trainable=False))
    circ.add(gates.CZ(0, 1))
    # Grover diffusion operator
    circ.add(gates.RZ(0, np.pi / 2, trainable=False))
    circ.add(gates.RX(0, np.pi / 2, trainable=False))
    circ.add(gates.RZ(0, np.pi / 2, trainable=False))
    circ.add(gates.RX(0, np.pi, trainable=False))
    circ.add(gates.RX(1, np.pi, trainable=False))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.RX(0, np.pi, trainable=False))
    circ.add(gates.RX(1, np.pi, trainable=False))
    circ.add(gates.RZ(0, np.pi / 2, trainable=False))
    circ.add(gates.RX(0, np.pi / 2, trainable=False))
    circ.add(gates.RZ(0, np.pi / 2, trainable=False))
    circ.add(gates.RZ(1, np.pi / 2, trainable=False))
    circ.add(gates.RX(1, np.pi / 2, trainable=False))
    circ.add(gates.RZ(1, np.pi / 2, trainable=False))
    return circ


def qft_circuit() -> Circuit:
    """Return a 3-qubit Quantum Fourier Transform circuit.

    The circuit is expressed in the RX / RZ / CZ primitive gate basis so that
    it can be processed by :class:`CircuitEncoder` and fed into the RL agent.

    Returns:
        Qibo :class:`Circuit` with ``density_matrix=True``.
    """
    circ = Circuit(3, density_matrix=True)
    # Initial Hadamard layer
    circ.add(gates.RZ(0, np.pi / 2, trainable=False))
    circ.add(gates.RZ(1, np.pi / 2, trainable=False))
    circ.add(gates.RZ(2, np.pi / 2, trainable=False))
    circ.add(gates.RX(0, np.pi / 2, trainable=False))
    circ.add(gates.RX(1, np.pi / 2, trainable=False))
    circ.add(gates.RX(2, np.pi / 2, trainable=False))
    circ.add(gates.RZ(0, np.pi / 2, trainable=False))
    circ.add(gates.RZ(1, np.pi / 2, trainable=False))
    circ.add(gates.RZ(2, 3 * np.pi / 2, trainable=False))
    # Controlled phase rotations
    circ.add(gates.CZ(1, 2))
    circ.add(gates.RX(1, -np.pi / 4, trainable=False))
    circ.add(gates.CZ(1, 2))
    circ.add(gates.RX(1, np.pi / 4, trainable=False))
    circ.add(gates.RZ(2, np.pi / 8, trainable=False))
    circ.add(gates.RZ(1, np.pi / 4, trainable=False))
    circ.add(gates.CZ(0, 2))
    circ.add(gates.RX(0, -np.pi / 8, trainable=False))
    circ.add(gates.CZ(0, 2))
    circ.add(gates.RX(0, np.pi / 8, trainable=False))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.RX(0, -np.pi / 4, trainable=False))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.RX(0, -np.pi / 4, trainable=False))
    return circ
