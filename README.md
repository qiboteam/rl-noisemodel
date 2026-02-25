# RL-NoiseModel

**Quantum noise modeling through reinforcement learning**

A refactored, modular package for generating quantum circuit datasets with custom noise models and training reinforcement learning agents in gymnasium environments.

## 🐍 Requirements

**Python >=3.10, <3.14** (due to qibo dependency constraints)

If you have Python 3.14, please create a virtual environment with Python 3.10-3.13:
```bash
# Using conda
conda create -n rlnoise python=3.11
conda activate rlnoise
poetry install

# Using pyenv  
pyenv install 3.11.0
pyenv local 3.11.0
poetry install
```

## Features

### Dataset Generation
- 🎯 **Clean API**: Simple, intuitive interface for dataset generation
- 🔧 **Flexible Configuration**: Use Python objects or JSON for configuration
- 📊 **Multiple Dataset Types**: Training, evaluation, and randomized benchmarking
- 🎲 **Circuit Variety**: Clifford and non-Clifford circuits with arbitrary depths
- 💾 **Easy I/O**: Save and load datasets in standard formats

### Gymnasium Environment
- 🎮 **RL-Ready**: Gymnasium-compatible environment for training agents
- 🎯 **Flexible Rewards**: 4 distance metrics × 4 transform functions
- 🔍 **Observation Space**: Sliding window over circuit encoding
- ⚙️ **Action Space**: 4 noise parameters per qubit (Pauli X/Z, reset, depolarizing)
- 📈 **Train/Val Split**: Automatic dataset splitting

### Development
- ✅ **Well Tested**: 80+ comprehensive unit tests
- 📝 **Type Safe**: Pydantic models for configuration validation
- 📚 **Examples**: Interactive Jupyter notebooks and Python scripts

## Installation

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/qiboteam/rl-noisemodel.git
cd rl-noisemodel

# Install with poetry
poetry install

# Run tests
poetry run pytest
```

### Using pip

```bash
pip install -e .
```

## Quick Start

### Dataset Generation

```python
from rlnoise import DatasetConfig, NoiseConfig, DatasetGenerator

# Configure dataset
dataset_config = DatasetConfig(
    n_circuits=100,
    qubits=2,
    moments=10,
    clifford=True,
)

# Configure noise model
noise_config = NoiseConfig(
    primitive_gates=["rx", "rz", "cz"],
    dep_lambda=0.02,
    p0=0.03,
)

# Generate dataset
generator = DatasetGenerator(dataset_config, noise_config)
dataset = generator.generate()

# Save to disk
dataset.save("my_dataset")

# Load later
from rlnoise import CircuitDataset
loaded_dataset = CircuitDataset.load("my_dataset.npz")
```

### Gymnasium Environment

```python
from rlnoise import (
    create_quantum_circuit_env,
    GymEnvConfig,
    RewardConfig,
)

# Create environment from dataset
env_config = GymEnvConfig(kernel_size=3, val_split=0.2)
reward_config = RewardConfig(metric="trace", function="inverted", alpha=20.0)

env = create_quantum_circuit_env(
    dataset=dataset,
    primitive_gates=["rx", "rz", "cz"],
    env_config=env_config,
    reward_config=reward_config,
)

# Standard gym loop
obs, info = env.reset()
terminated = False
while not terminated:
    action = env.action_space.sample()  # Or use your policy
    obs, reward, terminated, truncated, info = env.step(action)
```

## Documentation

### Configuration

**DatasetConfig** - Controls circuit generation:
- `n_circuits`: Number of circuits to generate
- `qubits`: Number of qubits per circuit
- `moments`: Circuit depth (number of gate layers)
- `clifford`: Use Clifford gates with quantized angles
- `mixed`: Mix random and Clifford circuits
- `eval_size`: Number of circuits for evaluation set
- `eval_depth`: Circuit depth for evaluation

**NoiseConfig** - Defines noise model:
- `primitive_gates`: List of gate types ["rx", "rz", "cz"]
- `dep_lambda`: Depolarizing noise strength (0-1)
- `p0`: Amplitude damping probability (0-1)
- `epsilon_x`: Coherent X rotation error
- `epsilon_z`: Coherent Z rotation error
- `x_coherent_on_gate`: Gates to apply X errors to
- `z_coherent_on_gate`: Gates to apply Z errors to
- `damping_on_gate`: Gates to apply damping to
- `depol_on_gate`: Gates to apply depolarizing noise to

**GymEnvConfig** - Gymnasium environment settings:
- `kernel_size`: Sliding window size (must be odd, default=3)
- `action_penalty`: Penalty per action taken (default=0.0)
- `action_space_max_value`: Maximum noise parameter value (default=0.06)
- `enable_only_depolarizing`: Restrict to depolarizing noise only (default=False)
- `val_split`: Validation set fraction (default=0.2)

**RewardConfig** - Reward function settings:
- `metric`: Distance metric ("mse", "mae", "trace", "fidelity")
- `function`: Transform function ("log", "linear", "inverted", "inverted_squared")
- `alpha`: Scaling factor (default=20.0)
- `epsilon`: Numerical stability (default=1e-10)

### Dataset Generation

```python
# Basic dataset
dataset = generator.generate()

# Evaluation dataset (different depth)
eval_dataset = generator.generate_evaluation_set(
    eval_depth=20,
    eval_size=50
)

# Randomized benchmarking datasets
rb_datasets = generator.generate_rb_dataset(
    start=3,
    stop=30,
    step=3,
    n_circuits_per_depth=50
)
```

### Working with Datasets

```python
# Dataset properties
print(len(dataset))                    # Number of circuits
print(dataset.shape)                   # Shape of circuit array
circuit, label = dataset[0]            # Get single sample

# Train/val split
train_dataset, val_dataset = dataset.split(val_fraction=0.2)

# Save and load
dataset.save("path/to/dataset")
loaded = CircuitDataset.load("path/to/dataset.npz")
```

## Examples

### Dataset Generation
See [examples/dataset_generation_example.ipynb](examples/dataset_generation_example.ipynb) for:
- Basic dataset generation
- Multi-qubit circuits
- Evaluation datasets
- Randomized benchmarking
- Dataset visualization
- JSON configuration

### Gymnasium Environment
See [examples/gym_environment_example.ipynb](examples/gym_environment_example.ipynb) for:
- Creating environments from datasets
- Understanding observation and action spaces
- Performing different types of actions (random, targeted, mixed)
- Running complete episodes
- Working with validation circuits
- Custom reward functions
- Multi-qubit environments
- Action strategies

## Project Structure

```
rl-noisemodel/
├── src/rlnoise/          # Source code
│   ├── __init__.py
│   ├── config.py         # Pydantic configuration models
│   ├── dataset.py        # Main dataset classes
│   ├── circuit_generator.py  # Circuit generation
│   ├── circuit_encoder.py    # Circuit encoding for ML
│   └── noise_model.py    # Noise application
├── tests/                # Unit tests
│   ├── test_config.py
│   ├── test_circuit_generator.py
│   ├── test_noise_model.py
│   ├── test_circuit_encoder.py
│   └── test_dataset.py
├── examples/             # Example notebooks and scripts
│   └── dataset_generation_example.ipynb
├── experiments/          # Experiment configurations and results
├── old/                  # Legacy code (archived)
├── pyproject.toml        # Poetry configuration
└── README.md
```

## Compatibility with Old Code

The refactored code maintains compatibility with the old dataset format. Datasets generated with the new code have the same structure:

```python
# Old style (still works via JSON config)
from rlnoise.config import ExperimentConfig

config_dict = {
    "dataset": {"n_circuits": 100, "qubits": 1, "moments": 10},
    "noise": {"dep_lambda": 0.02, "p0": 0.03}
}
config = ExperimentConfig.from_json(config_dict)
generator = DatasetGenerator.from_config(config)
```

## Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=rlnoise --cov-report=html

# Run specific test file
poetry run pytest tests/test_dataset.py
```

## Development

```bash
# Install with dev dependencies
poetry install --with dev

# Format code
poetry run black src/ tests/

# Sort imports
poetry run isort src/ tests/

# Type checking
poetry run mypy src/
```

## Migration Guide

### From Old Code

**Old:**
```python
from rlnoise.dataset import Dataset

dataset = Dataset(config_file)
dataset.save(filename)
```

**New:**
```python
from rlnoise import DatasetConfig, NoiseConfig, DatasetGenerator

dataset_config = DatasetConfig(n_circuits=100, qubits=1, moments=10)
noise_config = NoiseConfig(dep_lambda=0.02, p0=0.03)

generator = DatasetGenerator(dataset_config, noise_config)
dataset = generator.generate()
dataset.save(filename)
```

### Benefits of New Code

- ✅ **Modular**: Each component has a single responsibility
- ✅ **Typed**: Pydantic provides validation and IDE support
- ✅ **Testable**: Clean separation makes testing easier
- ✅ **Documented**: Comprehensive docstrings and examples
- ✅ **Maintainable**: Clear structure and naming conventions

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

Apache License 2.0

## Authors

- Simone Bordoni
- Piergiorgio Buttarini
- Andrea Papaluca
- Alejandro Sopena

## Citation

If you use this package in your research, please cite:

```bibtex
@software{rlnoise,
  title = {RL-NoiseModel: Quantum noise modeling through reinforcement learning},
  author = {Bordoni, Simone and Buttarini, Piergiorgio and Papaluca, Andrea and Sopena, Alejandro},
  url = {https://github.com/qiboteam/rl-noisemodel},
  year = {2024}
}
```
