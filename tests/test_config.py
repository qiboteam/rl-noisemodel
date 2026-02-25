"""Unit tests for configuration models."""

import pytest
from rlnoise.config import (
    NoiseConfig,
    DatasetConfig,
    RandomizedBenchmarkingConfig,
    ExperimentConfig,
)


class TestNoiseConfig:
    """Test NoiseConfig validation and creation."""
    
    def test_default_config(self):
        """Test default noise configuration."""
        config = NoiseConfig()
        
        assert config.primitive_gates == ["rx", "rz"]
        assert config.dep_lambda == 0.02
        assert config.p0 == 0.03
        assert config.epsilon_x == 0.04
        assert config.epsilon_z == 0.02
    
    def test_custom_config(self):
        """Test custom noise parameters."""
        config = NoiseConfig(
            primitive_gates=["rx", "rz", "cz"],
            dep_lambda=0.05,
            p0=0.01,
        )
        
        assert "cz" in config.primitive_gates
        assert config.dep_lambda == 0.05
        assert config.p0 == 0.01
    
    def test_gate_name_normalization(self):
        """Test that gate names are normalized to lowercase."""
        config = NoiseConfig(
            primitive_gates=["RX", "RZ", "CZ"],
            x_coherent_on_gate=["RX"],
        )
        
        assert config.primitive_gates == ["rx", "rz", "cz"]
        assert config.x_coherent_on_gate == ["rx"]
    
    def test_invalid_lambda(self):
        """Test validation of depolarizing parameter."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            NoiseConfig(dep_lambda=1.5)
    
    def test_invalid_p0(self):
        """Test validation of reset parameter."""
        with pytest.raises(Exception):
            NoiseConfig(p0=-0.1)


class TestDatasetConfig:
    """Test DatasetConfig validation and creation."""
    
    def test_default_config(self):
        """Test default dataset configuration."""
        config = DatasetConfig()
        
        assert config.n_circuits == 100
        assert config.moments == 10
        assert config.qubits == 1
        assert config.clifford is True
    
    def test_custom_config(self):
        """Test custom dataset parameters."""
        config = DatasetConfig(
            n_circuits=500,
            qubits=3,
            moments=20,
            clifford=False,
        )
        
        assert config.n_circuits == 500
        assert config.qubits == 3
        assert config.moments == 20
        assert config.clifford is False
    
    def test_invalid_n_circuits(self):
        """Test validation of circuit count."""
        with pytest.raises(Exception):
            DatasetConfig(n_circuits=0)
    
    def test_invalid_qubits(self):
        """Test validation of qubit count."""
        with pytest.raises(Exception):
            DatasetConfig(qubits=-1)


class TestRandomizedBenchmarkingConfig:
    """Test RB configuration."""
    
    def test_default_config(self):
        """Test default RB configuration."""
        config = RandomizedBenchmarkingConfig()
        
        assert config.start == 3
        assert config.stop == 31
        assert config.step == 3
        assert config.n_circ == 50
    
    def test_custom_config(self):
        """Test custom RB parameters."""
        config = RandomizedBenchmarkingConfig(
            start=5,
            stop=50,
            step=5,
            n_circ=100,
        )
        
        assert config.start == 5
        assert config.stop == 50
        assert config.step == 5
        assert config.n_circ == 100


class TestExperimentConfig:
    """Test complete experiment configuration."""
    
    def test_from_json(self):
        """Test creating config from JSON dictionary."""
        json_dict = {
            "dataset": {
                "n_circuits": 200,
                "qubits": 2,
                "moments": 15,
            },
            "noise": {
                "primitive_gates": ["rx", "rz", "cz"],
                "dep_lambda": 0.03,
            },
            "rb": {
                "start": 5,
                "stop": 30,
                "step": 5,
            }
        }
        
        config = ExperimentConfig.from_json(json_dict)
        
        assert config.dataset.n_circuits == 200
        assert config.dataset.qubits == 2
        assert config.noise.dep_lambda == 0.03
        assert config.rb is not None
        assert config.rb.start == 5
    
    def test_from_json_without_rb(self):
        """Test config creation without RB parameters."""
        json_dict = {
            "dataset": {
                "n_circuits": 100,
            },
            "noise": {
                "dep_lambda": 0.02,
            }
        }
        
        config = ExperimentConfig.from_json(json_dict)
        
        assert config.dataset.n_circuits == 100
        assert config.rb is None
