"""Configuration models for RL Noise."""

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class GateSpecificNoise(BaseModel):
    """Configuration for gate-specific noise application.

    Attributes:
        gate: Gate name to apply noise to (e.g., 'rx', 'rz', 'cz')
        noise_channel: Type of noise channel
        noise_parameter: Noise parameter value (float for uniform, list for per-qubit).
            This feature is available only for coherent errors.
        angle_dependent: For coherent errors on rotation gates, scale by gate angle
    """

    gate: str
    noise_channel: Literal["depolarizing", "damping", "coherent_z", "coherent_x"]
    noise_parameter: Union[float, List[float]]
    angle_dependent: bool = Field(default=False)

    @field_validator("gate")
    @classmethod
    def validate_gate_name(cls, v):
        """Ensure gate name is lowercase."""
        return v.lower()

    @model_validator(mode='after')
    def validate_angle_dependent(self):
        """Validate that angle_dependent is only used with coherent errors."""
        if self.angle_dependent:
            if self.noise_channel not in ["coherent_x", "coherent_z"]:
                raise ValueError(
                    "angle_dependent can only be used with coherent_x or coherent_z"
                )
        return self

    def validate_parameter_length(self, qubits: int):
        """Validate that list parameters match the number of qubits.

        Args:
            qubits: Number of qubits to validate against

        Raises:
            ValueError: If list parameter has incorrect length
        """
        if isinstance(self.noise_parameter, list):
            if len(self.noise_parameter) != qubits:
                raise ValueError(
                    f"noise_parameter list length ({len(self.noise_parameter)}) must match "
                    f"number of qubits ({qubits})"
                )

    def get_parameter_list(self, qubits: int) -> List[float]:
        """Get noise parameter as list per qubit.

        Args:
            qubits: Number of qubits

        Returns:
            List of noise parameters with length qubits
        """
        if isinstance(self.noise_parameter, list):
            return self.noise_parameter
        return [self.noise_parameter] * qubits


class NoiseConfig(BaseModel):
    """Configuration for noise model parameters.

    The noise configuration is now organized as a list of gate-specific noise
    specifications, providing more flexibility and clarity in defining which
    noise channels apply to which gates.

    Attributes:
        noise_list: List of gate-specific noise configurations
    """

    noise_list: List[GateSpecificNoise] = Field(default_factory=list)

    def validate_list_lengths(self, qubits: int):
        """Validate that all list parameters match the number of qubits.

        Args:
            qubits: Number of qubits to validate against

        Raises:
            ValueError: If any list parameter has incorrect length
        """
        for noise in self.noise_list:
            noise.validate_parameter_length(qubits)

    def get_noise_for_gate(self, gate: str) -> List[GateSpecificNoise]:
        """Get all noise configurations for a specific gate.

        Args:
            gate: Gate name to query

        Returns:
            List of GateSpecificNoise objects for the specified gate
        """
        return [noise for noise in self.noise_list if noise.gate.lower() == gate.lower()]

    def get_noise_by_channel(self, channel: str) -> List[GateSpecificNoise]:
        """Get all noise configurations for a specific channel type.

        Args:
            channel: Noise channel type to query

        Returns:
            List of GateSpecificNoise objects with the specified channel
        """
        return [noise for noise in self.noise_list if noise.noise_channel == channel]

    def get_gates_for_channel(self, channel: str) -> List[str]:
        """Get list of gates that have a specific noise channel applied.

        Args:
            channel: Noise channel type to query

        Returns:
            List of gate names
        """
        return [noise.gate for noise in self.noise_list if noise.noise_channel == channel]

    def __str__(self) -> str:
        """String representation showing key parameters."""
        if not self.noise_list:
            return (
                f"\n{'='*60}\n"
                f"  NoiseConfig\n"
                f"{'='*60}\n"
                f"  No noise configured\n"
                f"{'='*60}"
            )

        # Group by gate
        gate_noise_map: Dict[str, List[GateSpecificNoise]] = {}
        for noise in self.noise_list:
            if noise.gate not in gate_noise_map:
                gate_noise_map[noise.gate] = []
            gate_noise_map[noise.gate].append(noise)

        # Format output
        lines = [
            f"\n{'='*60}",
            "  NoiseConfig",
            f"{'='*60}",
            "  Gate-Specific Noise:"
        ]

        for gate in sorted(gate_noise_map.keys()):
            lines.append(f"    Gate: {gate}")
            for noise in gate_noise_map[gate]:
                # Format parameter
                if isinstance(noise.noise_parameter, list):
                    param_str = "[" + ", ".join(f"{v:.4f}" for v in noise.noise_parameter) + "]"
                else:
                    param_str = f"{noise.noise_parameter:.4f}"

                # Add angle-dependent indicator
                angle_dep = " (angle-dependent)" if noise.angle_dependent else ""
                lines.append(f"      â€¢ {noise.noise_channel}: {param_str}{angle_dep}")

        lines.append(f"{'='*60}")
        return "\n".join(lines)


class DatasetConfig(BaseModel):
    """Configuration for dataset generation.

    Attributes:
        n_circuits: Number of circuits to generate
        moments: Number of moments (circuit depth)
        qubits: Number of qubits in circuits
        primitive_gates: List of primitive gate names (e.g., ['rx', 'rz', 'cz'])
        distributed_clifford: Use distributed Clifford gates
        clifford: Generate Clifford circuits (quantized angles)
        mixed: Mix random and Clifford circuits
    """

    n_circuits: int = Field(default=100, gt=0)
    moments: int = Field(default=10, gt=0)
    qubits: int = Field(default=1, gt=0)
    primitive_gates: List[str] = Field(default=["rx", "rz"])
    distributed_clifford: bool = Field(default=False)
    clifford: bool = Field(default=True)
    mixed: bool = Field(default=False)

    @field_validator("primitive_gates")
    @classmethod
    def validate_gate_names(cls, v):
        """Ensure gate names are lowercase."""
        return [gate.lower() for gate in v]

    @model_validator(mode='after')
    def validate_config(self):
        """Validate configuration consistency."""
        # Check CZ gate only with multiple qubits
        if "cz" in self.primitive_gates and self.qubits < 2:
            raise ValueError("CZ gate requires at least 2 qubits")
        if "cnot" in self.primitive_gates and self.qubits < 2:
            raise ValueError("CNOT gate requires at least 2 qubits")

        return self

    def __str__(self) -> str:
        """String representation showing key parameters."""
        circuit_type = "Clifford" if self.clifford else "Arbitrary"
        if self.mixed:
            circuit_type = "Mixed (Clifford + Arbitrary)"

        gates_str = ", ".join(self.primitive_gates)

        return (
            f"\n{'='*50}\n"
            f"  DatasetConfig\n"
            f"{'='*50}\n"
            f"    â€¢ Circuits:       {self.n_circuits}\n"
            f"    â€¢ Qubits:         {self.qubits}\n"
            f"    â€¢ Moments:        {self.moments}\n"
            f"    â€¢ Type:           {circuit_type}\n"
            f"    â€¢ Gates:          [{gates_str}]\n"
            f"{'='*50}\n"
        )


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
        agent: Agent training configuration (optional)
        rb: Randomized benchmarking configuration (optional)
    """

    dataset: DatasetConfig
    noise: NoiseConfig
    gym_env: Optional[GymEnvConfig] = None
    reward: Optional[RewardConfig] = None
    agent: Optional["AgentConfig"] = None
    rb: Optional[RandomizedBenchmarkingConfig] = None

    @classmethod
    def from_json(cls, config_dict: dict) -> "ExperimentConfig":
        """Create configuration from a dictionary.

        Args:
            config_dict: Dictionary with configuration data.

        Returns:
            ExperimentConfig instance.
        """
        return cls(
            dataset=DatasetConfig(**config_dict.get("dataset", {})),
            noise=NoiseConfig(**config_dict.get("noise", {})),
            gym_env=GymEnvConfig(**config_dict["gym_env"]) if config_dict.get("gym_env") else None,
            reward=RewardConfig(**config_dict["reward"]) if config_dict.get("reward") else None,
            agent=AgentConfig(**config_dict["agent"]) if config_dict.get("agent") else None,
            rb=RandomizedBenchmarkingConfig(**config_dict["rb"]) if config_dict.get("rb") else None
        )

    def to_json_file(self, filepath: str) -> None:
        """Save configuration to a JSON file.

        Args:
            filepath: Destination path.  Parent directories are created
                automatically if they do not exist.
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(self.model_dump(), fh, indent=2)

    @classmethod
    def from_json_file(cls, filepath: str) -> "ExperimentConfig":
        """Load configuration from a JSON file previously saved with
        :meth:`to_json_file`.

        Args:
            filepath: Path to the JSON file.

        Returns:
            ExperimentConfig instance.
        """
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_json(data)


class AgentConfig(BaseModel):
    """Configuration for RL agent training.

    Attributes:
        policy: Policy type (e.g., 'MlpPolicy', 'CnnPolicy')
        features_dim: Dimension of feature extractor output
        filter_size: Size of CNN filter (width)
        n_filters: Number of CNN filters
        pi_net_arch: Policy network architecture (list of layer sizes)
        vf_net_arch: Value function network architecture (list of layer sizes)
        n_steps: Number of steps before PPO update
        batch_size: Batch size for PPO updates
        learning_rate: Learning rate for optimizer
        gamma: Discount factor
        clip_range: Clipping parameter for PPO
        verbose: Verbosity level (0=none, 1=info, 2=debug)
    """

    policy: str = Field(default="MlpPolicy")
    features_dim: int = Field(default=64, gt=0)
    filter_size: int = Field(default=3, gt=0)
    n_filters: int = Field(default=32, gt=0)
    pi_net_arch: List[int] = Field(default=[32, 32])
    vf_net_arch: List[int] = Field(default=[32, 32])
    n_steps: int = Field(default=2048, gt=0)
    batch_size: int = Field(default=64, gt=0)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    clip_range: float = Field(default=0.2, gt=0.0)
    verbose: int = Field(default=1, ge=0, le=2)

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v, info):
        """Ensure batch_size divides n_steps."""
        n_steps = info.data.get("n_steps", 2048)
        if n_steps % v != 0:
            raise ValueError(f"batch_size ({v}) must divide n_steps ({n_steps})")
        return v


# Resolve the forward reference to AgentConfig inside ExperimentConfig.
ExperimentConfig.model_rebuild()
