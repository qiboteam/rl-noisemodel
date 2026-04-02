# RL-NoiseModel

**Quantum noise modeling through reinforcement learning**

A modular Python package for generating quantum circuit datasets with custom noise models and training reinforcement learning agents in Gymnasium environments to learn and characterize quantum noise.

## Reference

This package accompanies the following publication:

> Simone Bordoni, Andrea Papaluca, Piergiorgio Buttarini, Alejandro Sopena, Stefano Giagu, Stefano Carrazza.
> **Quantum noise modeling through reinforcement learning.**
> *Quantum Science and Technology*, 2025.
> https://iopscience.iop.org/article/10.1088/2058-9565/ae1e98

The original implementation used to obtain the published results is preserved in the `old/` directory for reproducibility. The current package in `src/rlnoise/` is a refactored, modular version of that codebase.

## Authors

**Authors:** Simone Bordoni, Andrea Papaluca, Piergiorgio Buttarini, Alejandro Sopena

**Coordinators:** Stefano Giagu, Stefano Carrazza

## Requirements

**Python >=3.10, <3.14** (required by the `qibo` dependency)

If your default Python version is outside this range, create a virtual environment:

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
- Clean API for dataset generation with Pydantic-validated configuration
- Flexible noise model specification per gate and per qubit
- Support for Clifford and non-Clifford circuits with arbitrary depths
- Multiple dataset types: training, evaluation, and randomized benchmarking
- NumPy-based I/O for saving and loading datasets

### Gymnasium Environment
- Gymnasium-compatible environment for training RL agents
- Four distance metrics (MSE, MAE, trace distance, fidelity) and four reward transforms
- Sliding-window observation space over the circuit encoding
- Action space covering four noise parameters per qubit (coherent X/Z, reset, depolarizing)
- Automatic train/validation split

### Development
- 172 unit tests with 84% code coverage
- Pydantic models for configuration validation and type safety
- Interactive Jupyter notebook examples

## Installation

### Using Poetry (recommended)

```bash
git clone https://github.com/qiboteam/rl-noisemodel.git
cd rl-noisemodel
poetry install
poetry run pytest
```

### Using pip

```bash
pip install -e .
```

## Quick Start

### Dataset Generation

```python
from rlnoise import DatasetConfig, NoiseConfig, DatasetGenerator, GateSpecificNoise

# Configure dataset
dataset_config = DatasetConfig(
    n_circuits=100,
    qubits=2,
    moments=10,
    primitive_gates=["rx", "rz", "cz"],
    clifford=True,
)

# Configure noise model
noise_config = NoiseConfig(noise_list=[
    GateSpecificNoise(gate="rx", noise_channel="depolarizing", noise_parameter=0.02),
    GateSpecificNoise(gate="rx", noise_channel="damping", noise_parameter=0.03),
])

# Generate dataset
generator = DatasetGenerator(dataset_config, noise_config)
dataset = generator.generate()
dataset.save("my_dataset")

# Load later
from rlnoise import CircuitDataset
loaded_dataset = CircuitDataset.load("my_dataset.npz")
```

### Gymnasium Environment

```python
from rlnoise import create_quantum_circuit_env, GymEnvConfig, RewardConfig

env_config = GymEnvConfig(kernel_size=3, val_split=0.2)
reward_config = RewardConfig(metric="trace", function="inverted", alpha=20.0)

env = create_quantum_circuit_env(
    dataset=dataset,
    primitive_gates=["rx", "rz", "cz"],
    env_config=env_config,
    reward_config=reward_config,
)

obs, info = env.reset()
terminated = False
while not terminated:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

## Documentation

### Configuration

**DatasetConfig** -- circuit generation parameters:
- `n_circuits`: Number of circuits to generate
- `qubits`: Number of qubits per circuit
- `moments`: Circuit depth (gate layers)
- `primitive_gates`: List of gate type strings, e.g. `["rx", "rz", "cz"]`
- `clifford`: Use Clifford gates with quantized angles
- `mixed`: Mix random and Clifford circuits

**GateSpecificNoise** -- gate-level noise specification:
- `gate`: Gate name (e.g. `"rx"`, `"cz"`)
- `noise_channel`: One of `"depolarizing"`, `"damping"`, `"coherent_x"`, `"coherent_z"`
- `noise_parameter`: Scalar or per-qubit list of noise strengths
- `angle_dependent`: Scale coherent error by gate angle (coherent channels only)

**GymEnvConfig** -- environment parameters:
- `kernel_size`: Sliding window size (must be odd, default 3)
- `action_space_max_value`: Maximum noise parameter value (default 0.06)
- `enable_only_depolarizing`: Restrict to depolarizing noise only (default False)
- `val_split`: Validation set fraction (default 0.2)

**RewardConfig** -- reward function parameters:
- `metric`: Distance metric -- `"mse"`, `"mae"`, `"trace"`, or `"fidelity"`
- `function`: Transform function -- `"log"`, `"linear"`, `"inverted"`, or `"inverted_squared"`
- `alpha`: Scaling factor (default 20.0)

### Dataset Generation

```python
# Standard dataset
dataset = generator.generate()

# Randomized benchmarking datasets
rb_datasets = generator.generate_rb_dataset(
    start=3,
    stop=30,
    step=3,
    n_circuits_per_depth=50,
)
```

### Working with Datasets

```python
print(len(dataset))                          # Number of circuits
print(dataset.shape)                         # Shape of circuit array
circuit, label = dataset[0]                  # Get a single sample

train_dataset, val_dataset = dataset.split(val_fraction=0.2)

dataset.save("path/to/dataset")
loaded = CircuitDataset.load("path/to/dataset.npz")
```

## Examples

Interactive Jupyter notebooks are provided in the `examples/` directory:

- `dataset_generation.ipynb` -- dataset generation, multi-qubit circuits, and I/O
- `gym_environment.ipynb` -- Gymnasium environment usage and reward configuration
- `training.ipynb` -- RL agent training with Stable-Baselines3
- `benchmarking.ipynb` -- randomized benchmarking evaluation

## Project Structure

```
rl-noisemodel/
|-- src/rlnoise/               # Package source
|   |-- config.py              # Pydantic configuration models
|   |-- dataset.py             # Dataset classes
|   |-- circuit_generator.py   # Circuit generation
|   |-- circuit_encoder.py     # Circuit encoding for ML
|   |-- noise_model.py         # Noise application
|   |-- gym_env.py             # Gymnasium environment
|   |-- reward.py              # Reward functions
|   |-- neural_network.py      # CNN feature extractor
|   |-- callback.py            # Training callback
|   |-- rl_agent.py            # PPO-based RL agent
|   |-- benchmarking.py        # Randomized benchmarking
|   `-- visualization.py       # Plotting utilities
|-- tests/                     # Unit tests
|-- examples/                  # Jupyter notebooks
|-- experiments/               # Experiment scripts
|-- old/                       # Original implementation (archived)
|-- pyproject.toml
`-- README.md
```

## Original Implementation

The `old/` directory contains the original implementation that was used to produce the results reported in the accompanying publication. It is preserved for reproducibility and as a reference. The current package in `src/rlnoise/` is a refactored version with improved modularity, test coverage, and documentation.

## Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage report
poetry run pytest --cov=rlnoise --cov-report=html

# Run specific test file
poetry run pytest tests/test_dataset.py
```

## Development

```bash
# Install with development dependencies
poetry install --with dev

# Format code
poetry run black src/ tests/

# Sort imports
poetry run isort src/ tests/

# Lint
poetry run pylint src/rlnoise/
```

## License

Apache License 2.0

## Citation

If you use this package in your research, please cite:

```bibtex
@article{bordoni2025quantum,
  title   = {Quantum noise modeling through reinforcement learning},
  author  = {Bordoni, Simone and Papaluca, Andrea and Buttarini, Piergiorgio
             and Sopena, Alejandro and Giagu, Stefano and Carrazza, Stefano},
  journal = {Quantum Science and Technology},
  year    = {2025},
  doi     = {10.1088/2058-9565/ae1e98},
  url     = {https://iopscience.iop.org/article/10.1088/2058-9565/ae1e98}
}
```