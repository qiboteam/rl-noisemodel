"""Dataset generation for quantum circuit noise modeling."""

import os
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from rlnoise.config import DatasetConfig, NoiseConfig, ExperimentConfig
from rlnoise.circuit_generator import CircuitGenerator
from rlnoise.circuit_encoder import CircuitEncoder
from rlnoise.noise_model import QuantumNoiseModel


class CircuitDataset:
    """General-purpose container for quantum circuit dataset with labels.

    This class can be used for both training and evaluation datasets.

    Attributes:
        circuits: Array of circuit encodings, shape (n_circuits, n_moments, n_qubits, encoding_dim)
        labels: Array of density matrices from noisy circuits,
            shape (n_circuits, 2^n_qubits, 2^n_qubits)
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

    def __str__(self) -> str:
        """String representation showing dataset information."""
        n_circuits = len(self)
        # Support both dense (n, moments, qubits, enc) and object (n,) arrays
        first = self.circuits[0]
        n_moments, n_qubits, encoding_dim = first.shape
        shape_str = (
            self.circuits.shape[1:] if self.circuits.ndim == 4
            else f"variable (sample: {first.shape})"
        )

        result = (
            f"\n{'='*60}\n"
            f"  CircuitDataset\n"
            f"{'='*60}\n"
            f"    â€¢ Circuits:            {n_circuits}\n"
            f"    â€¢ Qubits:              {n_qubits}\n"
            f"    â€¢ Moments (depth):     {n_moments}\n"
            f"    â€¢ Encoding dimension:  {encoding_dim}\n"
            f"    â€¢ Circuit shape:       {shape_str}\n"
            f"{'='*60}"
        )

        return result

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
        noise_config: Configuration for noise model (optional, uses defaults if not provided)

    Example:
        >>> dataset_config = DatasetConfig(
        ...     n_circuits=100, qubits=2, moments=10, primitive_gates=["rx", "rz"])
        >>> noise_config = NoiseConfig(depolarizing=0.02, damping=0.03)
        >>> generator = DatasetGenerator(dataset_config, noise_config)
        >>> dataset = generator.generate()
        >>> dataset.save("my_dataset")
    """

    def __init__(self, dataset_config: DatasetConfig, noise_config: Optional[NoiseConfig] = None):
        self.dataset_config = dataset_config
        self.noise_config = noise_config if noise_config is not None else NoiseConfig()

        # Validate noise config against dataset config
        self._validate_configs()

        # Initialize components
        self.circuit_generator = CircuitGenerator(dataset_config)
        self.noise_model = QuantumNoiseModel(self.noise_config, dataset_config.qubits)
        self.encoder = CircuitEncoder(dataset_config.primitive_gates)

    def _validate_configs(self):
        """Validate that noise config is compatible with dataset config."""
        # Check that all gates in noise config exist in primitive gates
        primitive_gate_set = set(self.dataset_config.primitive_gates + ["none"])

        # Get all gates mentioned in noise_list
        for noise_spec in self.noise_config.noise_list:
            if noise_spec.gate.lower() not in primitive_gate_set:
                raise ValueError(
                    f"Gate '{noise_spec.gate}' in noise configuration is not in primitive_gates: "
                    f"{self.dataset_config.primitive_gates}"
                )

        # Validate list lengths for per-qubit noise
        self.noise_config.validate_list_lengths(self.dataset_config.qubits)

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
            CircuitDataset containing the generated circuits and labels
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
            print("Dataset generated.")

        dataset = CircuitDataset(
            circuits=encoded_circuits,
            labels=labels,
            config=self.dataset_config,
        )

        return dataset

    def generate_rb_dataset(  # pylint: disable=too-many-arguments,too-many-positional-arguments
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
                primitive_gates=self.dataset_config.primitive_gates,
                clifford=True,  # RB uses Clifford circuits
                distributed_clifford=True,
                mixed=False,
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
