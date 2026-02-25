"""Configuration models for RL Noise."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class NoiseConfig(BaseModel):
    """Configuration for noise model parameters.
    
    Attributes:
        primitive_gates: List of primitive gate names (e.g., ['rx', 'rz', 'cz'])
        channels: List of noise channel types
        dep_lambda: Depolarizing channel parameter (0-1)
        p0: Reset channel parameter (0-1)
        epsilon_x: Coherent X error parameter
        epsilon_z: Coherent Z error parameter
        x_coherent_on_gate: Gates to apply X coherent errors on
        z_coherent_on_gate: Gates to apply Z coherent errors on
        damping_on_gate: Gates to apply damping (reset) on
        depol_on_gate: Gates to apply depolarizing noise on
    """
    
    primitive_gates: List[str] = Field(default=["rx", "rz"])
    channels: List[str] = Field(default=["DepolarizingChannel", "ResetChannel"])
    dep_lambda: float = Field(default=0.02, ge=0.0, le=1.0)
    p0: float = Field(default=0.03, ge=0.0, le=1.0)
    epsilon_x: float = Field(default=0.04)
    epsilon_z: float = Field(default=0.02)
    x_coherent_on_gate: List[str] = Field(default=["rx"])
    z_coherent_on_gate: List[str] = Field(default=["rz"])
    damping_on_gate: List[str] = Field(default=["rx"])
    depol_on_gate: List[str] = Field(default=["rz"])

    @field_validator("primitive_gates", "x_coherent_on_gate", "z_coherent_on_gate", 
                     "damping_on_gate", "depol_on_gate")
    @classmethod
    def validate_gate_names(cls, v):
        """Ensure gate names are lowercase."""
        return [gate.lower() for gate in v]


class DatasetConfig(BaseModel):
    """Configuration for dataset generation.
    
    Attributes:
        n_circuits: Number of circuits to generate
        eval_size: Number of circuits for evaluation dataset
        eval_depth: Circuit depth for evaluation dataset
        moments: Number of moments (circuit depth) for training
        qubits: Number of qubits in circuits
        distributed_clifford: Use distributed Clifford gates
        clifford: Generate Clifford circuits (quantized angles)
        mixed: Mix random and Clifford circuits
    """
    
    n_circuits: int = Field(default=100, gt=0)
    eval_size: int = Field(default=100, gt=0)
    eval_depth: int = Field(default=15, gt=0)
    moments: int = Field(default=10, gt=0)
    qubits: int = Field(default=1, gt=0)
    distributed_clifford: bool = Field(default=False)
    clifford: bool = Field(default=True)
    mixed: bool = Field(default=False)


class RandomizedBenchmarkingConfig(BaseModel):
    """Configuration for randomized benchmarking experiments.
    
    Attributes:
        start: Starting circuit depth
        stop: Ending circuit depth  
        step: Step size for circuit depth
        n_circ: Number of circuits per depth
    """
    
    start: int = Field(default=3, gt=0)
    stop: int = Field(default=31, gt=0)
    step: int = Field(default=3, gt=0)
    n_circ: int = Field(default=50, gt=0)


class GymEnvConfig(BaseModel):
    """Configuration for gymnasium environment.
    
    Attributes:
        kernel_size: Size of sliding window for observations (must be odd)
        action_penalty: Penalty for applying noise actions
        action_space_max_value: Maximum value for action space
        enable_only_depolarizing: Only allow depolarizing noise actions
        val_split: Fraction of dataset to use for validation
    """
    
    kernel_size: int = Field(default=3, gt=0)
    action_penalty: float = Field(default=0.0, ge=0.0)
    action_space_max_value: float = Field(default=0.06, gt=0.0)
    enable_only_depolarizing: bool = Field(default=False)
    val_split: float = Field(default=0.2, ge=0.0, le=1.0)
    
    @field_validator("kernel_size")
    @classmethod
    def validate_kernel_size(cls, v):
        """Ensure kernel size is odd."""
        if v % 2 == 0:
            raise ValueError("kernel_size must be an odd number")
        return v


class RewardConfig(BaseModel):
    """Configuration for reward function.
    
    Attributes:
        metric: Distance metric ('mse', 'fidelity', 'trace', 'mae')
        function: Reward function type ('log', 'linear', 'inverted', 'inverted_squared')
        alpha: Scaling parameter for reward function
    """
    
    metric: Literal["mse", "fidelity", "trace", "mae"] = Field(default="trace")
    function: Literal["log", "linear", "inverted", "inverted_squared"] = Field(
        default="inverted_squared"
    )
    alpha: float = Field(default=20.0, gt=0.0)


class ExperimentConfig(BaseModel):
    """Complete experiment configuration.
    
    Attributes:
        dataset: Dataset generation configuration
        noise: Noise model configuration
        gym_env: Gym environment configuration (optional)
        reward: Reward function configuration (optional)
        rb: Randomized benchmarking configuration (optional)
    """
    
    dataset: DatasetConfig
    noise: NoiseConfig
    gym_env: Optional[GymEnvConfig] = None
    reward: Optional[RewardConfig] = None
    rb: Optional[RandomizedBenchmarkingConfig] = None

    @classmethod
    def from_json(cls, config_dict: dict) -> "ExperimentConfig":
        """Create configuration from JSON dictionary."""
        return cls(
            dataset=DatasetConfig(**config_dict.get("dataset", {})),
            noise=NoiseConfig(**config_dict.get("noise", {})),
            gym_env=GymEnvConfig(**config_dict["gym_env"]) if "gym_env" in config_dict else None,
            reward=RewardConfig(**config_dict["reward"]) if "reward" in config_dict else None,
            rb=RandomizedBenchmarkingConfig(**config_dict["rb"]) if "rb" in config_dict else None
        )
