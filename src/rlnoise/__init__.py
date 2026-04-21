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
    grover_circuit,
    qft_circuit,
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
    plot_rb_decay,
    plot_shots,
    plot_density_matrix_heatmap,
    plot_circuit_metrics,
)
from rlnoise.benchmarking import (
    generate_rb_circuits,
    fit_rb_decay,
    evaluate_benchmarks,
    evaluate_circuit,
    evaluate_on_dataset,
    maximally_mixed_state,
    summarize_benchmarks,
    summarize_rb_parameters,
    summarize_circuit_metrics,
)
from rlnoise.analysis import (
    collect_actions,
    noise_summary,
    plot_noise_distributions,
    plot_noise_by_gate,
    plot_noise_by_qubit,
    plot_spatial_noise,
    plot_spatial_noise_per_qubit,
    plot_noise_correlation,
    plot_mean_noise_per_gate,
    NOISE_CHANNELS,
    NOISE_LABELS,
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
    # Circuit generation
    "grover_circuit",
    "qft_circuit",
    # Visualization
    "plot_training_dashboard",
    "plot_benchmarking_results",
    "plot_rb_decay",
    "plot_shots",
    "plot_density_matrix_heatmap",
    "plot_circuit_metrics",
    # Benchmarking
    "generate_rb_circuits",
    "fit_rb_decay",
    "evaluate_benchmarks",
    "evaluate_circuit",
    "evaluate_on_dataset",
    "maximally_mixed_state",
    "summarize_benchmarks",
    "summarize_rb_parameters",
    "summarize_circuit_metrics",
    # Analysis / Explainability
    "collect_actions",
    "noise_summary",
    "plot_noise_distributions",
    "plot_noise_by_gate",
    "plot_noise_by_qubit",
    "plot_spatial_noise",
    "plot_spatial_noise_per_qubit",
    "plot_noise_correlation",
    "plot_mean_noise_per_gate",
    "NOISE_CHANNELS",
    "NOISE_LABELS",
]
