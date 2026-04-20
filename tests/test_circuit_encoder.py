"""Unit tests for circuit encoder."""

import pytest
import numpy as np
from qibo import gates
from qibo.models import Circuit

from rlnoise.circuit_encoder import CircuitEncoder


class TestCircuitEncoder:
    """Test CircuitEncoder functionality."""
    
    @pytest.fixture
    def encoder(self):
        """Create encoder with standard gates."""
        return CircuitEncoder(primitive_gates=["rx", "rz", "cz"])
    
    @pytest.fixture
    def simple_circuit(self):
        """Create a simple test circuit."""
        circuit = Circuit(2, density_matrix=True)
        circuit.add(gates.RX(0, theta=np.pi/2))
        circuit.add(gates.RZ(1, theta=np.pi))
        circuit.add(gates.CZ(0, 1))
        return circuit
    
    def test_initialization(self):
        """Test encoder initialization."""
        encoder = CircuitEncoder(primitive_gates=["rx", "rz"])
        
        assert encoder.encoding_dim == 8
        assert "rx" in encoder.primitive_gates
        assert "rz" in encoder.primitive_gates
    
    def test_gate_to_idx(self, encoder):
        """Test gate type to index mapping."""
        assert encoder._gate_to_idx(gates.RX) == encoder.IDX_RX
        assert encoder._gate_to_idx(gates.RZ) == encoder.IDX_RZ
        assert encoder._gate_to_idx(gates.CZ) == encoder.IDX_CZ
        assert encoder._gate_to_idx("param") == encoder.IDX_PARAM
    
    def test_gate_to_idx_invalid(self, encoder):
        """Test that invalid gate types raise error."""
        with pytest.raises(ValueError):
            encoder._gate_to_idx("invalid_gate")
    
    def test_gate_to_array_rx(self, encoder):
        """Test encoding of RX gate."""
        gate = gates.RX(0, theta=np.pi)
        encoding = encoder.gate_to_array(gate, qubit=0)
        
        assert encoding.shape == (8,)
        assert encoding[encoder.IDX_RX] == 1
        assert np.isclose(encoding[encoder.IDX_PARAM], 0.5)  # π / 2π
    
    def test_gate_to_array_rz(self, encoder):
        """Test encoding of RZ gate."""
        gate = gates.RZ(0, theta=np.pi/2)
        encoding = encoder.gate_to_array(gate, qubit=0)
        
        assert encoding[encoder.IDX_RZ] == 1
        assert np.isclose(encoding[encoder.IDX_PARAM], 0.25)  # π/2 / 2π
    
    def test_gate_to_array_cz(self, encoder):
        """Test encoding of CZ gate."""
        gate = gates.CZ(0, 1)
        
        # Control qubit encoding
        encoding_control = encoder.gate_to_array(gate, qubit=0)
        # Target qubit encoding
        encoding_target = encoder.gate_to_array(gate, qubit=1)
        
        # One should be 1, one should be -1
        assert abs(encoding_control[encoder.IDX_CZ]) == 1
        assert abs(encoding_target[encoder.IDX_CZ]) == 1
    
    def test_gate_to_array_none(self, encoder):
        """Test encoding of identity (None)."""
        encoding = encoder.gate_to_array(None, qubit=0)
        
        assert np.allclose(encoding, np.zeros(8))
    
    def test_circuit_to_array(self, encoder, simple_circuit):
        """Test encoding of complete circuit."""
        array = encoder.circuit_to_array(simple_circuit)
        
        # Check shape: (n_moments, n_qubits, encoding_dim)
        assert array.shape[0] == len(simple_circuit.queue.moments)
        assert array.shape[1] == 2  # 2 qubits
        assert array.shape[2] == 8  # encoding dimension
    
    def test_array_to_gate_rx(self, encoder):
        """Test decoding RX gate."""
        # Create encoding
        encoding = np.zeros(8)
        encoding[encoder.IDX_RX] = 1
        encoding[encoder.IDX_PARAM] = 0.25  # π/2 / 2π
        
        gate, channels = encoder.array_to_gate(encoding, qubit=0)
        
        assert type(gate) == gates.RX
        assert np.isclose(gate.init_kwargs["theta"], np.pi/2)
        assert len(channels) == 0
    
    def test_array_to_gate_with_noise(self, encoder):
        """Test decoding gate with noise channels."""
        encoding = np.zeros(8)
        encoding[encoder.IDX_RZ] = 1
        encoding[encoder.IDX_PARAM] = 0.5  # π / 2π
        encoding[encoder.IDX_DEPOL] = 0.02
        encoding[encoder.IDX_RESET] = 0.01
        
        gate, channels = encoder.array_to_gate(encoding, qubit=0)
        
        assert type(gate) == gates.RZ
        assert len(channels) == 2  # Depolarizing + Reset
    
    def test_roundtrip_single_qubit(self, encoder):
        """Test encoding and decoding roundtrip for single-qubit circuit."""
        original_circuit = Circuit(1, density_matrix=True)
        original_circuit.add(gates.RX(0, theta=np.pi/4))
        original_circuit.add(gates.RZ(0, theta=3*np.pi/2))
        
        # Encode
        array = encoder.circuit_to_array(original_circuit)
        
        # Decode
        reconstructed_circuit = encoder.array_to_circuit(array)
        
        # Check structure
        assert reconstructed_circuit.nqubits == original_circuit.nqubits
        assert len(reconstructed_circuit.queue.moments) == len(original_circuit.queue.moments)
    
    def test_roundtrip_two_qubit(self, encoder, simple_circuit):
        """Test encoding and decoding roundtrip for two-qubit circuit."""
        # Encode
        array = encoder.circuit_to_array(simple_circuit)
        
        # Decode
        reconstructed_circuit = encoder.array_to_circuit(array)
        
        # Check structure
        assert reconstructed_circuit.nqubits == simple_circuit.nqubits
    
    def test_empty_circuit(self, encoder):
        """Test encoding empty circuit."""
        circuit = Circuit(2, density_matrix=True)
        array = encoder.circuit_to_array(circuit)

        # Qibo may report 0 or 1 empty moments for an empty circuit
        assert array.ndim >= 1
        # Any gates present should be zero-encoded
        if array.shape[0] > 0:
            assert np.allclose(array[:, :, :3], 0.0)  # no gate flags set
    
    def test_single_qubit_add_gate(self, encoder):
        """Test adding single-qubit gate to circuit."""
        circuit = Circuit(2, density_matrix=True)
        encoding = np.zeros(8)
        encoding[encoder.IDX_RX] = 1
        encoding[encoder.IDX_PARAM] = 0.25
        
        encoder._add_single_qubit_gate(circuit, encoding, qubit=0)
        
        assert len(circuit.queue) == 1
        assert type(circuit.queue[0]) == gates.RX
    
    def test_two_qubit_add_gate(self, encoder):
        """Test adding two-qubit gate to circuit."""
        circuit = Circuit(2, density_matrix=True)
        
        # Create moment with CZ gate
        moment = np.zeros((2, 8))
        moment[0, encoder.IDX_CZ] = 1  # Control
        moment[1, encoder.IDX_CZ] = -1  # Target
        
        encoder._add_two_qubit_gate(circuit, moment, qubits=[0, 1])
        
        # Should have CZ gate
        has_cz = any(type(gate) == gates.CZ for gate in circuit.queue)
        assert has_cz

    def test_array_to_circuit_mixed_moment_cz_and_single_qubit(self, encoder):
        """Regression test: CZ gate and single-qubit gate in the same moment.

        When CZ(0,2) and RX(1) occupy the same circuit moment, array_to_circuit
        must emit both the two-qubit gate AND the single-qubit gate.
        Previously only the CZ pair was processed and RX(1) was silently dropped.
        """
        enc3 = CircuitEncoder(primitive_gates=["rx", "rz", "cz"])
        circuit = Circuit(3, density_matrix=True)
        circuit.add(gates.CZ(0, 2))   # qubits 0,2 occupied in moment 0
        circuit.add(gates.RX(1, 0.5)) # qubit 1 free -> placed in same moment 0

        # Confirm qibo really puts them in the same moment
        assert len(circuit.queue.moments) == 1, "Expected CZ and RX to share one moment"

        arr = enc3.circuit_to_array(circuit)
        assert arr.shape[0] == 1  # one moment

        reconstructed = enc3.array_to_circuit(arr)
        gate_types = [type(g) for g in reconstructed.queue]

        assert gates.CZ in gate_types, "CZ gate missing from reconstructed circuit"
        assert gates.RX in gate_types, "RX gate missing from reconstructed circuit (regression)"

    def test_array_to_circuit_mixed_moment_fidelity(self, encoder):
        """Regression test: circuit reconstructed from mixed moment achieves high fidelity.

        The density matrix of a circuit with CZ(0,2)+RX(1) in the same moment,
        encoded and decoded through array_to_circuit, must be identical to the
        original (within floating-point noise).
        """
        from qibo.quantum_info import fidelity as qibo_fidelity

        enc3 = CircuitEncoder(primitive_gates=["rx", "rz", "cz"])
        circuit = Circuit(3, density_matrix=True)
        circuit.add(gates.CZ(0, 2))
        circuit.add(gates.RX(1, 0.9))
        circuit.add(gates.RZ(0, 0.4))

        arr = enc3.circuit_to_array(circuit)
        reconstructed = enc3.array_to_circuit(arr)

        dm_original = circuit().state()
        dm_reconstructed = reconstructed().state()

        fid = float(qibo_fidelity(dm_original, dm_reconstructed))
        assert fid > 0.9999, f"Reconstructed circuit fidelity {fid:.6f} too low (expected >0.9999)"
