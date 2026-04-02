"""RL Noise Modeling - Quantum noise modeling through reinforcement learning."""

__version__ = "0.1.0"

from rlnoise.config import (
    DatasetConfig,
    NoiseConfig,
    GateSpecificNoise,
    GymEnvConfig,
    RewardConfig,
    ExperimentConfig,
    AgentConfig,
)
from rlnoise.dataset import (
    CircuitDataset,
    DatasetGenerator,
)
from rlnoise.circuit_generator import (
    CircuitGenerator,
)
from rlnoise.circuit_encoder import (
    CircuitEncoder,
)
from rlnoise.noise_model import (
    QuantumNoiseModel,
)
from rlnoise.gym_env import (
    QuantumCircuitEnv,
    create_quantum_circuit_env,
)
from rlnoise.reward import (
    RewardFunction,
    create_reward_function,
)
from rlnoise.neural_network import (
    CNNFeaturesExtractor,
)
from rlnoise.callback import (
    TrainingCallback,
)
from rlnoise.rl_agent import (
    RLAgent,
)
from rlnoise.visualization import (
    plot_training_dashboard,
    plot_benchmarking_results,
)
from rlnoise.benchmarking import (
    generate_rb_circuits,
    fit_rb_decay,
    evaluate_benchmarks,
    maximally_mixed_state,
)

__all__ = [
    "__version__",
    # Configuration
    "DatasetConfig",
    "NoiseConfig",
    "GateSpecificNoise",
    "GymEnvConfig",
    "RewardConfig",
    "ExperimentConfig",
    "AgentConfig",
    # Dataset
    "CircuitDataset",
    "DatasetGenerator",
    # Circuit Generation and Encoding
    "CircuitGenerator",
    "CircuitEncoder",
    # Noise Model
    "QuantumNoiseModel",
    # Gym Environment
    "QuantumCircuitEnv",
    # Reward
    "RewardFunction",
    "create_reward_function",
    # Training
    "CNNFeaturesExtractor",
    "TrainingCallback",
    "RLAgent",
    # Visualization
    "plot_training_dashboard",
    "plot_benchmarking_results",
    # Benchmarking
    "generate_rb_circuits",
    "fit_rb_decay",
    "evaluate_benchmarks",
    "maximally_mixed_state",
]
