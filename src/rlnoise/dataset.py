"""Dataset generation for quantum circuit noise modeling."""

import os
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from qibo.models import Circuit

from rlnoise.config import DatasetConfig, NoiseConfig, ExperimentConfig
from rlnoise.circuit_generator import CircuitGenerator
from rlnoise.circuit_encoder import CircuitEncoder
from rlnoise.noise_model import QuantumNoiseModel


class CircuitDataset:
    """Container for quantum circuit dataset with labels.
    
    Attributes:
        circuits: Array of circuit encodings, shape (n_circuits, n_moments, n_qubits, encoding_dim)
        labels: Array of density matrices from noisy circuits, shape (n_circuits, 2^n_qubits, 2^n_qubits)
        config: Dataset configuration used to generate the data
    """
    
    def __init__(
        self,
        circuits: np.ndarray,
        labels: np.ndarray,
        config: Optional[DatasetConfig] = None,
    ):
        self.circuits = circuits
        self.labels = labels
        self.config = config
    
    def __len__(self) -> int:
        """Return number of circuits in dataset."""
        return len(self.circuits)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a single circuit and label by index."""
        return self.circuits[idx], self.labels[idx]
    
    def save(self, filepath: str):
        """Save dataset to disk in .npz format.
        
        Args:
            filepath: Path to save file (without extension)
        """
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Add .npz extension if not present
        if not filepath.endswith('.npz'):
            filepath = filepath + '.npz'
        
        np.savez(
            filepath,
            circuits=self.circuits,
            labels=self.labels,
            allow_pickle=True
        )
    
    @classmethod
    def load(cls, filepath: str) -> "CircuitDataset":
        """Load dataset from disk.
        
        Args:
            filepath: Path to .npz file
            
        Returns:
            Loaded CircuitDataset
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        data = np.load(filepath, allow_pickle=True)
        circuits = data['circuits']
        labels = data['labels']
        
        return cls(circuits=circuits, labels=labels)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Return shape of circuit array."""
        return self.circuits.shape
    
    def split(self, val_fraction: float = 0.2) -> Tuple["CircuitDataset", "CircuitDataset"]:
        """Split dataset into training and validation sets.
        
        Args:
            val_fraction: Fraction of data to use for validation
            
        Returns:
            (train_dataset, val_dataset)
        """
        n_val = int(len(self) * val_fraction)
        n_train = len(self) - n_val
        
        # Create random permutation
        indices = np.random.permutation(len(self))
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        train_dataset = CircuitDataset(
            circuits=self.circuits[train_indices],
            labels=self.labels[train_indices],
            config=self.config
        )
        
        val_dataset = CircuitDataset(
            circuits=self.circuits[val_indices],
            labels=self.labels[val_indices],
            config=self.config
        )
        
        return train_dataset, val_dataset


class DatasetGenerator:
    """Generate datasets for training noise models.
    
    This is the main class for creating datasets. It coordinates:
    - Circuit generation (random/Clifford)
    - Noise application
    - Circuit encoding for ML
    - Label generation (density matrices)
    
    Args:
        dataset_config: Configuration for dataset generation
        noise_config: Configuration for noise model
    
    Example:
        >>> dataset_config = DatasetConfig(n_circuits=100, qubits=2, moments=10)
        >>> noise_config = NoiseConfig(dep_lambda=0.02, p0=0.03)
        >>> generator = DatasetGenerator(dataset_config, noise_config)
        >>> dataset = generator.generate()
        >>> dataset.save("my_dataset")
    """
    
    def __init__(self, dataset_config: DatasetConfig, noise_config: NoiseConfig):
        self.dataset_config = dataset_config
        self.noise_config = noise_config
        
        # Initialize components
        self.circuit_generator = CircuitGenerator(dataset_config, noise_config.primitive_gates)
        self.noise_model = QuantumNoiseModel(noise_config)
        self.encoder = CircuitEncoder(noise_config.primitive_gates)
    
    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "DatasetGenerator":
        """Create generator from complete experiment configuration.
        
        Args:
            config: Complete experiment configuration
            
        Returns:
            Configured DatasetGenerator
        """
        return cls(config.dataset, config.noise)
    
    def generate(self, verbose: bool = True) -> CircuitDataset:
        """Generate complete dataset with circuits and labels.
        
        Args:
            verbose: Print progress information
            
        Returns:
            CircuitDataset with encoded circuits and density matrix labels
        """
        if verbose:
            print(f"Generating {self.dataset_config.n_circuits} circuits...")
        
        # Generate circuits
        circuits = self.circuit_generator.generate_batch(
            n_circuits=self.dataset_config.n_circuits,
            mixed=self.dataset_config.mixed
        )
        
        if verbose:
            print("Applying noise model...")
        
        # Apply noise and compute density matrices
        noisy_circuits = [self.noise_model.apply(circuit) for circuit in circuits]
        labels = np.array([circ().state() for circ in noisy_circuits])
        
        if verbose:
            print("Encoding circuits...")
        
        # Encode circuits as arrays
        encoded_circuits = np.array([
            self.encoder.circuit_to_array(circuit)
            for circuit in circuits
        ], dtype=object)
        
        if verbose:
            print(f"Dataset generated: {len(circuits)} circuits, "
                  f"{self.dataset_config.qubits} qubits, "
                  f"{self.dataset_config.moments} moments")
        
        return CircuitDataset(
            circuits=encoded_circuits,
            labels=labels,
            config=self.dataset_config
        )
    
    def generate_evaluation_set(
        self,
        eval_depth: Optional[int] = None,
        eval_size: Optional[int] = None,
        verbose: bool = True
    ) -> CircuitDataset:
        """Generate evaluation dataset with different parameters.
        
        This is useful for testing generalization to different circuit depths.
        
        Args:
            eval_depth: Circuit depth for evaluation (uses config default if None)
            eval_size: Number of circuits (uses config default if None)
            verbose: Print progress information
            
        Returns:
            CircuitDataset for evaluation
        """
        # Use config defaults if not specified
        eval_depth = eval_depth or self.dataset_config.eval_depth
        eval_size = eval_size or self.dataset_config.eval_size
        
        if verbose:
            print(f"Generating evaluation set: {eval_size} circuits with depth {eval_depth}...")
        
        # Create temporary config with eval parameters
        eval_config = DatasetConfig(
            n_circuits=eval_size,
            moments=eval_depth,
            qubits=self.dataset_config.qubits,
            clifford=self.dataset_config.clifford,
            distributed_clifford=self.dataset_config.distributed_clifford,
            mixed=self.dataset_config.mixed
        )
        
        # Create temporary generator
        temp_generator = DatasetGenerator(eval_config, self.noise_config)
        
        return temp_generator.generate(verbose=verbose)
    
    def generate_rb_dataset(
        self,
        start: int,
        stop: int,
        step: int,
        n_circuits_per_depth: int,
        verbose: bool = True
    ) -> List[CircuitDataset]:
        """Generate dataset for randomized benchmarking experiments.
        
        Creates multiple datasets with increasing circuit depths.
        
        Args:
            start: Starting circuit depth
            stop: Ending circuit depth (exclusive)
            step: Step size between depths
            n_circuits_per_depth: Number of circuits to generate per depth
            verbose: Print progress information
            
        Returns:
            List of CircuitDatasets, one per depth level
        """
        datasets = []
        
        for depth in range(start, stop, step):
            if verbose:
                print(f"Generating RB dataset for depth {depth}...")
            
            # Create config for this depth
            rb_config = DatasetConfig(
                n_circuits=n_circuits_per_depth,
                moments=depth,
                qubits=self.dataset_config.qubits,
                clifford=True,  # RB uses Clifford circuits
                distributed_clifford=True,
                mixed=False
            )
            
            # Generate dataset
            temp_generator = DatasetGenerator(rb_config, self.noise_config)
            dataset = temp_generator.generate(verbose=False)
            datasets.append(dataset)
        
        if verbose:
            print(f"Generated {len(datasets)} RB datasets")
        
        return datasets


# Re-export main classes
__all__ = [
    "CircuitDataset",
    "DatasetGenerator",
    "DatasetConfig",
    "NoiseConfig",
    "ExperimentConfig",
]
