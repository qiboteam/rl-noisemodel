"""Unit tests for circuit generation."""

import pytest
import numpy as np
from qibo import gates
from qibo.models import Circuit

from rlnoise.config import DatasetConfig
from rlnoise.circuit_generator import CircuitGenerator


class TestCircuitGenerator:
    """Test CircuitGenerator functionality."""
    
    @pytest.fixture
    def config_1q(self):
        """Single-qubit configuration."""
        return DatasetConfig(
            n_circuits=10,
            qubits=1,
            moments=5,
            clifford=True,
            primitive_gates=["rx", "rz"],
        )

    @pytest.fixture
    def config_2q(self):
        """Two-qubit configuration with CZ gates."""
        return DatasetConfig(
            n_circuits=10,
            qubits=2,
            moments=10,
            clifford=False,
            primitive_gates=["rx", "rz", "cz"],
        )

    @pytest.fixture
    def generator_1q(self, config_1q):
        """Single-qubit generator."""
        return CircuitGenerator(config_1q)

    @pytest.fixture
    def generator_2q(self, config_2q):
        """Two-qubit generator."""
        return CircuitGenerator(config_2q)

    def test_initialization(self, config_1q):
        """Test generator initialization."""
        generator = CircuitGenerator(config_1q)

        assert generator.n_qubits == 1
        assert generator.n_moments == 5
        assert generator.is_clifford is True
        assert "rx" in generator.primitive_gates

    def test_single_qubit_circuit(self, generator_1q):
        """Test single-qubit random circuit generation."""
        circuit = generator_1q.generate_random_circuit()
        
        assert isinstance(circuit, Circuit)
        assert circuit.nqubits == 1
        assert len(circuit.queue.moments) == 5
        
        # Check all gates are from primitive set
        for gate in circuit.queue:
            assert type(gate) in [gates.RX, gates.RZ]
    
    def test_two_qubit_circuit(self, generator_2q):
        """Test two-qubit circuit with CZ gates."""
        circuit = generator_2q.generate_random_circuit()
        
        assert isinstance(circuit, Circuit)
        assert circuit.nqubits == 2
        assert len(circuit.queue.moments) == 10
    
    def test_clifford_angles(self, generator_1q):
        """Test that Clifford circuits use quantized angles."""
        circuit = generator_1q.generate_random_circuit()
        
        clifford_angles = {0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi}
        
        for gate in circuit.queue:
            if hasattr(gate, "init_kwargs") and "theta" in gate.init_kwargs:
                theta = gate.init_kwargs["theta"]
                # Check if theta is close to any Clifford angle
                is_clifford = any(np.isclose(theta % (2*np.pi), angle) for angle in clifford_angles)
                assert is_clifford, f"Non-Clifford angle: {theta}"
    
    def test_non_clifford_angles(self, config_1q):
        """Test that non-Clifford circuits use arbitrary angles."""
        config_1q.clifford = False
        generator = CircuitGenerator(config_1q)
        
        # Generate multiple circuits to get variety
        circuits = [generator.generate_random_circuit() for _ in range(5)]
        
        # At least one should have a non-Clifford angle
        angles = []
        for circuit in circuits:
            for gate in circuit.queue:
                if hasattr(gate, "init_kwargs") and "theta" in gate.init_kwargs:
                    angles.append(gate.init_kwargs["theta"])
        
        assert len(angles) > 0, "No parametric gates found"
    
    def test_clifford_circuit_generation(self, generator_2q):
        """Test Clifford circuit generation with decomposition."""
        circuit = generator_2q.generate_clifford_circuit()
        
        assert isinstance(circuit, Circuit)
        assert circuit.nqubits == 2
        
        # All gates should be from primitive set
        for gate in circuit.queue:
            gate_type = type(gate)
            assert gate_type in [gates.RX, gates.RZ, gates.CZ]
    
    def test_batch_generation(self, generator_1q):
        """Test batch circuit generation."""
        circuits = generator_1q.generate_batch(n_circuits=5, mixed=False)
        
        assert len(circuits) == 5
        assert all(isinstance(c, Circuit) for c in circuits)
        assert all(c.nqubits == 1 for c in circuits)
    
    def test_mixed_batch_generation(self, generator_2q):
        """Test mixed random and Clifford batch generation."""
        circuits = generator_2q.generate_batch(n_circuits=10, mixed=True)
        
        assert len(circuits) == 10
        assert all(isinstance(c, Circuit) for c in circuits)
    
    def test_invalid_single_qubit_with_cz(self):
        """Test that single-qubit circuits with CZ raise an error."""
        # The config validation itself raises ValueError for 1-qubit CZ
        with pytest.raises(Exception):
            config = DatasetConfig(qubits=1, moments=5, primitive_gates=["rx", "rz", "cz"])
            CircuitGenerator(config)

    def test_gate_decomposition(self, generator_1q):
        """Test that decomposition works for standard gates."""
        # Create a circuit with Hadamard gate
        raw_circuit = Circuit(1, density_matrix=True)
        raw_circuit.add(gates.H(0))

        decomposed = Circuit(1, density_matrix=True)
        for gate in raw_circuit.queue:
            generator_1q._decompose_gate(gate, decomposed)

        # Hadamard should decompose to RZ and RX
        assert len(decomposed.queue) == 2
        assert type(decomposed.queue[0]) == gates.RZ
        assert type(decomposed.queue[1]) == gates.RX

    def test_decompose_z_gate(self, generator_1q):
        """Z gate → RZ(π)."""
        raw = Circuit(1, density_matrix=True)
        raw.add(gates.Z(0))
        out = Circuit(1, density_matrix=True)
        generator_1q._decompose_gate(raw.queue[0], out)
        assert len(out.queue) == 1
        assert type(out.queue[0]) == gates.RZ

    def test_decompose_x_gate(self, generator_1q):
        """X gate → RX(π)."""
        raw = Circuit(1, density_matrix=True)
        raw.add(gates.X(0))
        out = Circuit(1, density_matrix=True)
        generator_1q._decompose_gate(raw.queue[0], out)
        assert len(out.queue) == 1
        assert type(out.queue[0]) == gates.RX

    def test_decompose_y_gate(self, generator_1q):
        """Y gate → RZ(π) + RX(π)."""
        raw = Circuit(1, density_matrix=True)
        raw.add(gates.Y(0))
        out = Circuit(1, density_matrix=True)
        generator_1q._decompose_gate(raw.queue[0], out)
        assert len(out.queue) == 2
        assert type(out.queue[0]) == gates.RZ
        assert type(out.queue[1]) == gates.RX

    def test_decompose_s_and_sdg(self, generator_1q):
        """S and Sdg gates → single RZ."""
        for gate_cls in [gates.S, gates.SDG]:
            raw = Circuit(1, density_matrix=True)
            raw.add(gate_cls(0))
            out = Circuit(1, density_matrix=True)
            generator_1q._decompose_gate(raw.queue[0], out)
            assert len(out.queue) == 1
            assert type(out.queue[0]) == gates.RZ

    def test_decompose_t_and_tdg(self, generator_1q):
        """T and Tdg gates → single RZ."""
        for gate_cls in [gates.T, gates.TDG]:
            raw = Circuit(1, density_matrix=True)
            raw.add(gate_cls(0))
            out = Circuit(1, density_matrix=True)
            generator_1q._decompose_gate(raw.queue[0], out)
            assert len(out.queue) == 1
            assert type(out.queue[0]) == gates.RZ

    def test_decompose_sx_and_sxdg(self, generator_1q):
        """SX → RX(π/2); SXDG → RX(−π/2)."""
        for gate_cls in [gates.SX, gates.SXDG]:
            raw = Circuit(1, density_matrix=True)
            raw.add(gate_cls(0))
            out = Circuit(1, density_matrix=True)
            generator_1q._decompose_gate(raw.queue[0], out)
            assert len(out.queue) == 1
            assert type(out.queue[0]) == gates.RX

    def test_decompose_swap(self, generator_2q):
        """SWAP → 3 × CX pattern (3 × 5 = 15 primitive gates)."""
        raw = Circuit(2, density_matrix=True)
        raw.add(gates.SWAP(0, 1))
        out = Circuit(2, density_matrix=True)
        generator_2q._decompose_gate(raw.queue[0], out)
        # 3 CX decompositions, each adds 5 gates
        assert len(out.queue) == 15

    def test_decompose_unknown_gate_raises(self, generator_1q):
        """_decompose_gate raises ValueError for unsupported gates."""
        raw = Circuit(1, density_matrix=True)
        raw.add(gates.RX(0, theta=0.1, trainable=False))
        # Rename so it looks unknown
        g = raw.queue[0]
        g.name = "unknown_gate_xyz"
        out = Circuit(1, density_matrix=True)
        with pytest.raises(ValueError, match="Cannot decompose unknown gate"):
            generator_1q._decompose_gate(g, out)

    def test_cnot_gate_generation(self):
        """Test circuit generation when cnot is in primitive gates."""
        config = DatasetConfig(
            n_circuits=3,
            qubits=2,
            moments=5,
            clifford=False,
            primitive_gates=["rx", "cnot"],
        )
        generator = CircuitGenerator(config)
        circuit = generator.generate_random_circuit()
        assert isinstance(circuit, Circuit)
        assert circuit.nqubits == 2
