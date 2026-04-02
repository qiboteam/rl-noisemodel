"""Unit tests for configuration models."""

import json
import os
import tempfile
import pytest
from rlnoise.config import (
    NoiseConfig,
    GateSpecificNoise,
    DatasetConfig,
    RandomizedBenchmarkingConfig,
    AgentConfig,
    ExperimentConfig,
)


class TestGateSpecificNoise:
    """Test GateSpecificNoise validation and creation."""
    
    def test_basic_creation(self):
        """Test creating a basic gate-specific noise."""
        noise = GateSpecificNoise(
            gate="rx",
            noise_channel="depolarizing",
            noise_parameter=0.05
        )
        
        assert noise.gate == "rx"
        assert noise.noise_channel == "depolarizing"
        assert noise.noise_parameter == 0.05
        assert noise.angle_dependent is False
    
    def test_gate_name_normalization(self):
        """Test that gate names are normalized to lowercase."""
        noise = GateSpecificNoise(
            gate="RX",
            noise_channel="coherent_x",
            noise_parameter=0.1
        )
        
        assert noise.gate == "rx"
    
    def test_per_qubit_parameters(self):
        """Test per-qubit noise parameters."""
        noise = GateSpecificNoise(
            gate="rx",
            noise_channel="coherent_x",
            noise_parameter=[0.1, 0.2, 0.3]
        )
        
        assert isinstance(noise.noise_parameter, list)
        assert len(noise.noise_parameter) == 3
    
    def test_angle_dependent_coherent(self):
        """Test angle-dependent coherent noise."""
        noise = GateSpecificNoise(
            gate="rx",
            noise_channel="coherent_x",
            noise_parameter=0.1,
            angle_dependent=True
        )
        
        assert noise.angle_dependent is True
    
    def test_angle_dependent_validation(self):
        """Test that angle_dependent only works with coherent errors."""
        with pytest.raises(ValueError, match="angle_dependent can only be used"):
            GateSpecificNoise(
                gate="rx",
                noise_channel="depolarizing",
                noise_parameter=0.1,
                angle_dependent=True
            )
    
    def test_parameter_list_conversion(self):
        """Test get_parameter_list method."""
        # Test with float
        noise1 = GateSpecificNoise(
            gate="rx",
            noise_channel="depolarizing",
            noise_parameter=0.05
        )
        assert noise1.get_parameter_list(3) == [0.05, 0.05, 0.05]
        
        # Test with list
        noise2 = GateSpecificNoise(
            gate="rx",
            noise_channel="coherent_x",
            noise_parameter=[0.1, 0.2]
        )
        assert noise2.get_parameter_list(2) == [0.1, 0.2]
    
    def test_validate_parameter_length(self):
        """Test parameter length validation."""
        noise = GateSpecificNoise(
            gate="rx",
            noise_channel="coherent_x",
            noise_parameter=[0.1, 0.2]
        )
        
        # Should pass
        noise.validate_parameter_length(2)
        
        # Should fail
        with pytest.raises(ValueError, match="noise_parameter list length"):
            noise.validate_parameter_length(3)


class TestNoiseConfig:
    """Test NoiseConfig validation and creation."""
    
    def test_empty_config(self):
        """Test empty noise configuration."""
        config = NoiseConfig()
        
        assert config.noise_list == []
    
    def test_with_noise_list(self):
        """Test noise config with gate-specific noise."""
        config = NoiseConfig(
            noise_list=[
                GateSpecificNoise(gate="rx", noise_channel="depolarizing", noise_parameter=0.05),
                GateSpecificNoise(gate="rz", noise_channel="coherent_z", noise_parameter=0.02),
            ]
        )
        
        assert len(config.noise_list) == 2
        assert config.noise_list[0].gate == "rx"
        assert config.noise_list[1].gate == "rz"
    
    def test_get_noise_for_gate(self):
        """Test querying noise by gate."""
        config = NoiseConfig(
            noise_list=[
                GateSpecificNoise(gate="rx", noise_channel="depolarizing", noise_parameter=0.05),
                GateSpecificNoise(gate="rx", noise_channel="coherent_x", noise_parameter=0.1),
                GateSpecificNoise(gate="rz", noise_channel="coherent_z", noise_parameter=0.02),
            ]
        )
        
        rx_noise = config.get_noise_for_gate("rx")
        assert len(rx_noise) == 2
        assert all(n.gate == "rx" for n in rx_noise)
        
        rz_noise = config.get_noise_for_gate("rz")
        assert len(rz_noise) == 1
    
    def test_get_noise_by_channel(self):
        """Test querying noise by channel type."""
        config = NoiseConfig(
            noise_list=[
                GateSpecificNoise(gate="rx", noise_channel="coherent_x", noise_parameter=0.1),
                GateSpecificNoise(gate="rz", noise_channel="coherent_z", noise_parameter=0.02),
                GateSpecificNoise(gate="rx", noise_channel="depolarizing", noise_parameter=0.05),
            ]
        )
        
        coherent_x = config.get_noise_by_channel("coherent_x")
        assert len(coherent_x) == 1
        assert coherent_x[0].noise_channel == "coherent_x"
    
    def test_get_gates_for_channel(self):
        """Test getting gates that have a specific noise channel."""
        config = NoiseConfig(
            noise_list=[
                GateSpecificNoise(gate="rx", noise_channel="coherent_x", noise_parameter=0.1),
                GateSpecificNoise(gate="rz", noise_channel="coherent_x", noise_parameter=0.1),
                GateSpecificNoise(gate="rx", noise_channel="depolarizing", noise_parameter=0.05),
            ]
        )
        
        gates = config.get_gates_for_channel("coherent_x")
        assert len(gates) == 2
        assert "rx" in gates
        assert "rz" in gates
    
    def test_validate_list_lengths(self):
        """Test validation of per-qubit parameter lengths."""
        config = NoiseConfig(
            noise_list=[
                GateSpecificNoise(gate="rx", noise_channel="coherent_x", noise_parameter=[0.1, 0.2]),
            ]
        )
        
        # Should pass
        config.validate_list_lengths(2)
        
        # Should fail
        with pytest.raises(ValueError):
            config.validate_list_lengths(3)


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
                "noise_list": [
                    {
                        "gate": "rx",
                        "noise_channel": "depolarizing",
                        "noise_parameter": 0.03
                    },
                    {
                        "gate": "rz",
                        "noise_channel": "coherent_z",
                        "noise_parameter": 0.02,
                        "angle_dependent": True
                    }
                ]
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
        assert len(config.noise.noise_list) == 2
        assert config.noise.noise_list[0].noise_parameter == 0.03
        assert config.rb is not None
        assert config.rb.start == 5
    
    def test_from_json_without_rb(self):
        """Test config creation without RB parameters."""
        json_dict = {
            "dataset": {
                "n_circuits": 100,
            },
            "noise": {
                "noise_list": []
            }
        }
        
        config = ExperimentConfig.from_json(json_dict)
        
        assert config.dataset.n_circuits == 100
        assert config.rb is None

    def test_from_json_with_agent(self):
        """Test config creation with agent key."""
        json_dict = {
            "dataset": {"n_circuits": 50},
            "noise": {"noise_list": []},
            "agent": {
                "policy": "MlpPolicy",
                "features_dim": 32,
                "filter_size": 2,
                "n_filters": 16,
                "pi_net_arch": [16],
                "vf_net_arch": [16],
                "n_steps": 32,
                "batch_size": 8,
            },
        }

        config = ExperimentConfig.from_json(json_dict)
        assert config.agent is not None
        assert config.agent.features_dim == 32

    def test_to_json_file_and_from_json_file(self):
        """Test round-trip JSON file serialization."""
        noise_config = NoiseConfig(
            noise_list=[
                GateSpecificNoise(gate="rx", noise_channel="depolarizing", noise_parameter=0.03),
            ]
        )
        dataset_config = DatasetConfig(n_circuits=10, qubits=1, moments=5)
        agent_config = AgentConfig(features_dim=32, n_steps=32, batch_size=8)

        exp = ExperimentConfig(
            dataset=dataset_config,
            noise=noise_config,
            agent=agent_config,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "subdir", "exp.json")
            exp.to_json_file(filepath)

            assert os.path.exists(filepath)

            reloaded = ExperimentConfig.from_json_file(filepath)

        assert reloaded.dataset.n_circuits == 10
        assert len(reloaded.noise.noise_list) == 1
        assert reloaded.agent is not None
        assert reloaded.agent.features_dim == 32


class TestConfigStrMethods:
    """Test __str__ methods for DatasetConfig and NoiseConfig."""

    def test_noise_config_str_empty(self):
        """Empty NoiseConfig __str__ mentions 'No noise configured'."""
        config = NoiseConfig()
        s = str(config)
        assert "NoiseConfig" in s
        assert "No noise configured" in s

    def test_noise_config_str_with_noise(self):
        """NoiseConfig __str__ lists gates and channels."""
        config = NoiseConfig(
            noise_list=[
                GateSpecificNoise(gate="rx", noise_channel="depolarizing", noise_parameter=0.05),
                GateSpecificNoise(gate="rz", noise_channel="coherent_z", noise_parameter=0.02,
                                  angle_dependent=True),
            ]
        )
        s = str(config)
        assert "NoiseConfig" in s
        assert "rx" in s
        assert "depolarizing" in s
        assert "angle-dependent" in s

    def test_noise_config_str_list_parameter(self):
        """NoiseConfig __str__ handles list noise parameters."""
        config = NoiseConfig(
            noise_list=[
                GateSpecificNoise(gate="rx", noise_channel="coherent_x",
                                  noise_parameter=[0.01, 0.02]),
            ]
        )
        s = str(config)
        assert "[" in s  # list representation

    def test_dataset_config_str(self):
        """DatasetConfig __str__ shows key fields."""
        config = DatasetConfig(n_circuits=50, qubits=2, moments=8, mixed=True)
        s = str(config)
        assert "DatasetConfig" in s
        assert "50" in s
        assert "Mixed" in s

    def test_dataset_config_cnot_validation(self):
        """DatasetConfig raises ValueError for cnot on 1 qubit."""
        with pytest.raises(Exception):
            DatasetConfig(qubits=1, primitive_gates=["rx", "cnot"])
