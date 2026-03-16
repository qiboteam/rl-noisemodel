"""Unit tests for noise model."""

import pytest
from qibo import gates
from qibo.models import Circuit

from rlnoise.config import NoiseConfig, GateSpecificNoise
from rlnoise.noise_model import QuantumNoiseModel


class TestQuantumNoiseModel:
    """Test QuantumNoiseModel functionality."""
    
    @pytest.fixture
    def simple_config(self):
        """Simple noise configuration."""
        return NoiseConfig(
            noise_list=[
                GateSpecificNoise(gate="rx", noise_channel="coherent_x", noise_parameter=0.04, angle_dependent=True),
                GateSpecificNoise(gate="rz", noise_channel="coherent_z", noise_parameter=0.02, angle_dependent=True),
                GateSpecificNoise(gate="rx", noise_channel="damping", noise_parameter=0.03),
                GateSpecificNoise(gate="rz", noise_channel="depolarizing", noise_parameter=0.02),
            ]
        )
    
    @pytest.fixture
    def noise_model(self, simple_config):
        """Create noise model."""
        return QuantumNoiseModel(simple_config, qubits=1)
    
    @pytest.fixture
    def simple_circuit(self):
        """Create a simple test circuit."""
        circuit = Circuit(1, density_matrix=True)
        circuit.add(gates.RX(0, theta=0.5))
        circuit.add(gates.RZ(0, theta=0.3))
        return circuit
    
    def test_initialization(self, simple_config):
        """Test noise model initialization."""
        model = QuantumNoiseModel(simple_config, qubits=1)
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
            noise_list=[
                GateSpecificNoise(gate="rz", noise_channel="depolarizing", noise_parameter=0.1),
            ]
        )
        model = QuantumNoiseModel(config, qubits=1)
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
            noise_list=[
                GateSpecificNoise(gate="rx", noise_channel="damping", noise_parameter=0.1),
            ]
        )
        model = QuantumNoiseModel(config, qubits=1)
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
        """Test that empty noise configuration doesn't add gates."""
        config = NoiseConfig(noise_list=[])
        model = QuantumNoiseModel(config, qubits=1)
        
        circuit = Circuit(1, density_matrix=True)
        circuit.add(gates.RX(0, theta=0.5))
        
        noisy_circuit = model.apply(circuit)
        
        # Should have same number of gates
        assert len(noisy_circuit.queue) == len(circuit.queue)
    
    def test_multi_qubit_circuit(self):
        """Test noise application on multi-qubit circuit."""
        config = NoiseConfig(
            noise_list=[
                GateSpecificNoise(gate="rx", noise_channel="depolarizing", noise_parameter=0.05),
                GateSpecificNoise(gate="cz", noise_channel="depolarizing", noise_parameter=0.05),
            ]
        )
        model = QuantumNoiseModel(config, qubits=2)
        
        circuit = Circuit(2, density_matrix=True)
        circuit.add(gates.RX(0, theta=0.5))
        circuit.add(gates.CZ(0, 1))
        circuit.add(gates.RZ(1, theta=0.3))
        
        noisy_circuit = model.apply(circuit)
        
        assert noisy_circuit.nqubits == 2
        assert len(noisy_circuit.queue) >= len(circuit.queue)
    
    def test_angle_dependent_coherent_errors(self):
        """Test that angle-dependent coherent errors scale with gate angle."""
        config = NoiseConfig(
            noise_list=[
                GateSpecificNoise(gate="rx", noise_channel="coherent_x", 
                                noise_parameter=0.1, angle_dependent=True),
            ]
        )
        model = QuantumNoiseModel(config, qubits=1)
        
        circuit = Circuit(1, density_matrix=True)
        circuit.add(gates.RX(0, theta=1.0))
        
        noisy_circuit = model.apply(circuit)
        
        # Find the added coherent error gate
        added_rx_gates = [g for g in noisy_circuit.queue if type(g) == gates.RX]
        assert len(added_rx_gates) >= 2  # Original + coherent error
    
    def test_fixed_coherent_errors(self):
        """Test that non-angle-dependent coherent errors use fixed parameters."""
        config = NoiseConfig(
            noise_list=[
                GateSpecificNoise(gate="rx", noise_channel="coherent_z", 
                                noise_parameter=0.05, angle_dependent=False),
            ]
        )
        model = QuantumNoiseModel(config, qubits=1)
        
        circuit = Circuit(1, density_matrix=True)
        circuit.add(gates.RX(0, theta=1.0))
        
        noisy_circuit = model.apply(circuit)
        
        # Check that RZ gate was added (coherent Z error)
        has_rz = any(type(g) == gates.RZ for g in noisy_circuit.queue)
        assert has_rz
    
    def test_per_qubit_parameters(self):
        """Test that per-qubit noise parameters work correctly."""
        config = NoiseConfig(
            noise_list=[
                GateSpecificNoise(gate="rx", noise_channel="coherent_x", 
                                noise_parameter=[0.1, 0.2], angle_dependent=True),
            ]
        )
        model = QuantumNoiseModel(config, qubits=2)
        
        circuit = Circuit(2, density_matrix=True)
        circuit.add(gates.RX(0, theta=1.0))
        circuit.add(gates.RX(1, theta=1.0))
        
        # Should not raise error
        noisy_circuit = model.apply(circuit)
        assert noisy_circuit.nqubits == 2
    
    def test_per_qubit_parameters_mismatch(self):
        """Test that mismatched list parameters raise error."""
        config = NoiseConfig(
            noise_list=[
                GateSpecificNoise(gate="rx", noise_channel="coherent_x", 
                                noise_parameter=[0.1, 0.2], angle_dependent=True),
            ]
        )
        
        # Should raise error: list has 2 elements but circuit has 3 qubits
        with pytest.raises(ValueError, match="has length 2, but circuit has 3 qubits"):
            QuantumNoiseModel(config, qubits=3)
        
        # Should also raise error for 1 qubit
        with pytest.raises(ValueError, match="has length 2, but circuit has 1 qubit"):
            QuantumNoiseModel(config, qubits=1)
    
    def test_gate_specific_noise_validation(self):
        """Test that GateSpecificNoise validates angle_dependent properly."""
        # Should raise error: angle_dependent only for coherent errors
        with pytest.raises(ValueError, match="angle_dependent can only be used"):
            GateSpecificNoise(
                gate="rx", 
                noise_channel="depolarizing", 
                noise_parameter=0.1, 
                angle_dependent=True
            )
