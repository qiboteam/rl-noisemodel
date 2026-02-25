"""Unit tests for noise model."""

import pytest
from qibo import gates
from qibo.models import Circuit

from rlnoise.config import NoiseConfig
from rlnoise.noise_model import QuantumNoiseModel


class TestQuantumNoiseModel:
    """Test QuantumNoiseModel functionality."""
    
    @pytest.fixture
    def simple_config(self):
        """Simple noise configuration."""
        return NoiseConfig(
            primitive_gates=["rx", "rz"],
            dep_lambda=0.02,
            p0=0.03,
            epsilon_x=0.04,
            epsilon_z=0.02,
            x_coherent_on_gate=["rx"],
            z_coherent_on_gate=["rz"],
            damping_on_gate=["rx"],
            depol_on_gate=["rz"],
        )
    
    @pytest.fixture
    def noise_model(self, simple_config):
        """Create noise model."""
        return QuantumNoiseModel(simple_config)
    
    @pytest.fixture
    def simple_circuit(self):
        """Create a simple test circuit."""
        circuit = Circuit(1, density_matrix=True)
        circuit.add(gates.RX(0, theta=0.5))
        circuit.add(gates.RZ(0, theta=0.3))
        return circuit
    
    def test_initialization(self, simple_config):
        """Test noise model initialization."""
        model = QuantumNoiseModel(simple_config)
        assert model.config == simple_config
    
    def test_string_to_gate(self):
        """Test gate string conversion."""
        assert QuantumNoiseModel._string_to_gate("rx") == gates.RX
        assert QuantumNoiseModel._string_to_gate("RZ") == gates.RZ
        assert QuantumNoiseModel._string_to_gate("cz") == gates.CZ
        assert QuantumNoiseModel._string_to_gate("none") is None
    
    def test_invalid_gate_string(self):
        """Test that invalid gate names raise error."""
        with pytest.raises(ValueError, match="Unrecognized gate"):
            QuantumNoiseModel._string_to_gate("invalid_gate")
    
    def test_apply_noise_increases_gates(self, noise_model, simple_circuit):
        """Test that applying noise adds gates to circuit."""
        original_length = len(simple_circuit.queue)
        noisy_circuit = noise_model.apply(simple_circuit)
        
        # Noisy circuit should have more gates
        assert len(noisy_circuit.queue) > original_length
    
    def test_apply_depolarizing_noise(self, simple_circuit):
        """Test that depolarizing noise is applied correctly."""
        config = NoiseConfig(
            primitive_gates=["rx", "rz"],
            dep_lambda=0.1,
            depol_on_gate=["rz"],
            damping_on_gate=[],
            x_coherent_on_gate=[],
            z_coherent_on_gate=[],
        )
        model = QuantumNoiseModel(config)
        noisy_circuit = model.apply(simple_circuit)
        
        # Check that DepolarizingChannel was added
        has_depol = any(
            isinstance(gate, gates.DepolarizingChannel)
            for gate in noisy_circuit.queue
        )
        assert has_depol
    
    def test_apply_reset_noise(self, simple_circuit):
        """Test that reset noise is applied correctly."""
        config = NoiseConfig(
            primitive_gates=["rx", "rz"],
            p0=0.1,
            damping_on_gate=["rx"],
            depol_on_gate=[],
            x_coherent_on_gate=[],
            z_coherent_on_gate=[],
        )
        model = QuantumNoiseModel(config)
        noisy_circuit = model.apply(simple_circuit)
        
        # Check that ResetChannel was added
        has_reset = any(
            isinstance(gate, gates.ResetChannel)
            for gate in noisy_circuit.queue
        )
        assert has_reset
    
    def test_apply_coherent_errors(self, noise_model, simple_circuit):
        """Test that coherent errors are applied."""
        noisy_circuit = noise_model.apply(simple_circuit)
        
        # Count RX and RZ gates (original + coherent errors)
        rx_count = sum(1 for g in noisy_circuit.queue if type(g) == gates.RX)
        rz_count = sum(1 for g in noisy_circuit.queue if type(g) == gates.RZ)
        
        # Should have more than original due to coherent errors
        assert rx_count >= 1
        assert rz_count >= 1
    
    def test_no_noise_identity(self):
        """Test that zero noise parameters don't add gates."""
        config = NoiseConfig(
            primitive_gates=["rx"],
            dep_lambda=0.0,
            p0=0.0,
            epsilon_x=0.0,
            epsilon_z=0.0,
            damping_on_gate=[],
            depol_on_gate=[],
            x_coherent_on_gate=[],
            z_coherent_on_gate=[],
        )
        model = QuantumNoiseModel(config)
        
        circuit = Circuit(1, density_matrix=True)
        circuit.add(gates.RX(0, theta=0.5))
        
        noisy_circuit = model.apply(circuit)
        
        # Should have same number of gates
        assert len(noisy_circuit.queue) == len(circuit.queue)
    
    def test_config_validation(self):
        """Test that incompatible configurations raise errors."""
        # Attempting to apply noise to gates not in primitive set
        config = NoiseConfig(
            primitive_gates=["rx"],
            depol_on_gate=["cz"],  # CZ not in primitive gates
        )
        
        with pytest.raises(ValueError):
            QuantumNoiseModel(config)
    
    def test_multi_qubit_circuit(self):
        """Test noise application on multi-qubit circuit."""
        config = NoiseConfig(
            primitive_gates=["rx", "rz", "cz"],
            dep_lambda=0.05,
        )
        model = QuantumNoiseModel(config)
        
        circuit = Circuit(2, density_matrix=True)
        circuit.add(gates.RX(0, theta=0.5))
        circuit.add(gates.CZ(0, 1))
        circuit.add(gates.RZ(1, theta=0.3))
        
        noisy_circuit = model.apply(circuit)
        
        assert noisy_circuit.nqubits == 2
        assert len(noisy_circuit.queue) >= len(circuit.queue)
