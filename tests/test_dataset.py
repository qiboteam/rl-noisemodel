"""Unit tests for dataset generation."""

import pytest
import numpy as np
import tempfile
import os

from rlnoise.config import DatasetConfig, NoiseConfig, GateSpecificNoise, ExperimentConfig
from rlnoise.dataset import CircuitDataset, DatasetGenerator


class TestCircuitDataset:
    """Test CircuitDataset functionality."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset."""
        # Create dummy data
        n_circuits = 10
        n_moments = 5
        n_qubits = 2
        encoding_dim = 8
        
        circuits = np.random.rand(n_circuits, n_moments, n_qubits, encoding_dim)
        labels = np.random.rand(n_circuits, 2**n_qubits, 2**n_qubits) + \
                 1j * np.random.rand(n_circuits, 2**n_qubits, 2**n_qubits)
        
        config = DatasetConfig(n_circuits=n_circuits, qubits=n_qubits, moments=n_moments)
        
        return CircuitDataset(circuits=circuits, labels=labels, config=config)
    
    def test_length(self, sample_dataset):
        """Test dataset length."""
        assert len(sample_dataset) == 10
    
    def test_getitem(self, sample_dataset):
        """Test getting items from dataset."""
        circuit, label = sample_dataset[0]
        
        assert circuit.shape == (5, 2, 8)
        assert label.shape == (4, 4)
    
    def test_shape(self, sample_dataset):
        """Test dataset shape property."""
        assert sample_dataset.shape == (10, 5, 2, 8)
    
    def test_save_and_load(self, sample_dataset):
        """Test saving and loading dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_dataset")
            
            # Save dataset
            sample_dataset.save(filepath)
            
            # Check file was created
            assert os.path.exists(filepath + ".npz")
            
            # Load dataset
            loaded_dataset = CircuitDataset.load(filepath + ".npz")
            
            # Verify data matches
            assert loaded_dataset.circuits.shape == sample_dataset.circuits.shape
            assert loaded_dataset.labels.shape == sample_dataset.labels.shape
    
    def test_split(self, sample_dataset):
        """Test train/val split."""
        train_dataset, val_dataset = sample_dataset.split(val_fraction=0.2)
        
        assert len(train_dataset) + len(val_dataset) == len(sample_dataset)
        assert len(val_dataset) == 2  # 20% of 10
        assert len(train_dataset) == 8  # 80% of 10
    
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            CircuitDataset.load("nonexistent_file.npz")


class TestDatasetGenerator:
    """Test DatasetGenerator functionality."""
    
    @pytest.fixture
    def simple_config(self):
        """Simple dataset configuration."""
        return DatasetConfig(
            n_circuits=5,
            qubits=1,
            moments=3,
            clifford=True,
        )
    
    @pytest.fixture
    def noise_config(self):
        """Simple noise configuration."""
        return NoiseConfig(
            noise_list=[
                GateSpecificNoise(gate="rx", noise_channel="depolarizing", noise_parameter=0.02),
                GateSpecificNoise(gate="rz", noise_channel="damping", noise_parameter=0.01),
            ]
        )
    
    @pytest.fixture
    def generator(self, simple_config, noise_config):
        """Create dataset generator."""
        return DatasetGenerator(simple_config, noise_config)
    
    def test_initialization(self, generator, simple_config, noise_config):
        """Test generator initialization."""
        assert generator.dataset_config == simple_config
        assert generator.noise_config == noise_config
    
    def test_from_config(self, simple_config, noise_config):
        """Test creating generator from ExperimentConfig."""
        exp_config = ExperimentConfig(dataset=simple_config, noise=noise_config)
        generator = DatasetGenerator.from_config(exp_config)
        
        assert generator.dataset_config == simple_config
        assert generator.noise_config == noise_config
    
    def test_generate_dataset(self, generator):
        """Test dataset generation."""
        dataset = generator.generate(verbose=False)
        
        assert isinstance(dataset, CircuitDataset)
        assert len(dataset) == 5
        assert dataset.labels.shape[0] == 5
    
    def test_dataset_labels_are_density_matrices(self, generator):
        """Test that labels are valid density matrices."""
        dataset = generator.generate(verbose=False)
        
        for i in range(len(dataset)):
            dm = dataset.labels[i]
            
            # Check shape
            assert dm.shape == (2, 2)  # 1 qubit -> 2x2 matrix
            
            # Check trace is close to 1
            assert np.abs(np.trace(dm) - 1.0) < 1e-6
    
    def test_generate_rb_dataset(self, generator):
        """Test randomized benchmarking dataset generation."""
        rb_datasets = generator.generate_rb_dataset(
            start=3,
            stop=9,
            step=3,
            n_circuits_per_depth=5,
            verbose=False
        )
        
        assert len(rb_datasets) == 2  # depths 3 and 6
        
        for dataset in rb_datasets:
            assert isinstance(dataset, CircuitDataset)
            assert len(dataset) == 5
    
    def test_multi_qubit_generation(self, noise_config):
        """Test dataset generation for multi-qubit circuits."""
        config = DatasetConfig(
            n_circuits=3,
            qubits=2,
            moments=5,
            primitive_gates=["rx", "rz", "cz"],
            clifford=True,
        )
        
        generator = DatasetGenerator(config, noise_config)
        
        dataset = generator.generate(verbose=False)
        
        assert len(dataset) == 3
        # 2 qubits -> 4x4 density matrices
        assert dataset.labels[0].shape == (4, 4)
    
    def test_mixed_dataset_generation(self, noise_config):
        """Test mixed random/Clifford dataset generation."""
        config = DatasetConfig(
            n_circuits=10,
            qubits=1,
            moments=5,
            clifford=True,
            mixed=True,
        )
        
        generator = DatasetGenerator(config, noise_config)
        dataset = generator.generate(verbose=False)
        
        assert len(dataset) == 10
    
    def test_non_clifford_generation(self, noise_config):
        """Test non-Clifford dataset generation."""
        config = DatasetConfig(
            n_circuits=5,
            qubits=1,
            moments=5,
            clifford=False,

        )
        
        generator = DatasetGenerator(config, noise_config)
        dataset = generator.generate(verbose=False)
        
        assert len(dataset) == 5
    
    def test_save_generated_dataset(self, generator):
        """Test generating and saving dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "generated_dataset")
            
            dataset = generator.generate(verbose=False)
            dataset.save(filepath)
            
            # Verify file exists
            assert os.path.exists(filepath + ".npz")
            
            # Load and verify
            loaded = CircuitDataset.load(filepath + ".npz")
            assert len(loaded) == len(dataset)
