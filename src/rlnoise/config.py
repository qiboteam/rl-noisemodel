"""Configuration models for RL Noise."""

from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class NoiseConfig(BaseModel):
    """Configuration for noise model parameters.
    
    Attributes:
        channels: List of noise channel types
        dep_lambda: Depolarizing channel parameter (float or list per qubit) [alias: depolarizing]
        p0: Reset channel parameter (float or list per qubit) [alias: damping]
        coherent_x: Coherent X error parameter (float or list per qubit) [alias: epsilon_x]
        coherent_y: Coherent Y error parameter (float or list per qubit) [alias: epsilon_z]
        x_coherent_on_gate: Gates to apply X coherent errors on
        z_coherent_on_gate: Gates to apply Z coherent errors on
        damping_on_gate: Gates to apply damping (reset) on
        depol_on_gate: Gates to apply depolarizing noise on
    """
    
    channels: List[str] = Field(default=["DepolarizingChannel", "ResetChannel"])
    dep_lambda: Union[float, List[float]] = Field(default=0.02, alias="depolarizing")
    p0: Union[float, List[float]] = Field(default=0.03, alias="damping")
    coherent_x: Union[float, List[float]] = Field(default=0.04, alias="epsilon_x")
    coherent_y: Union[float, List[float]] = Field(default=0.02, alias="epsilon_z")
    x_coherent_on_gate: List[str] = Field(default_factory=lambda: ["rx"])
    z_coherent_on_gate: List[str] = Field(default_factory=lambda: ["rz"])
    damping_on_gate: List[str] = Field(default_factory=lambda: ["rx"])
    depol_on_gate: List[str] = Field(default_factory=lambda: ["rz"])

    model_config = {"populate_by_name": True}

    @field_validator("x_coherent_on_gate", "z_coherent_on_gate", 
                     "damping_on_gate", "depol_on_gate")
    @classmethod
    def validate_gate_names(cls, v):
        """Ensure gate names are lowercase."""
        return [gate.lower() for gate in v]
    
    def validate_list_lengths(self, qubits: int):
        """Validate that list parameters match the number of qubits.
        
        Args:
            qubits: Number of qubits to validate against
            
        Raises:
            ValueError: If any list parameter has incorrect length
        """
        for field_name in ['dep_lambda', 'p0', 'coherent_x', 'coherent_y']:
            value = getattr(self, field_name)
            if isinstance(value, list):
                if len(value) != qubits:
                    raise ValueError(
                        f"{field_name} list length ({len(value)}) must match "
                        f"number of qubits ({qubits})"
                    )
    
    def _get_as_list(self, value: Union[float, List[float]], qubits: int) -> List[float]:
        """Convert float or list to list of length qubits.
        
        Args:
            value: Float or list of floats
            qubits: Number of qubits
            
        Returns:
            List of floats with length qubits
        """
        if isinstance(value, list):
            return value
        return [value] * qubits
    
    @property
    def depolarizing(self) -> Union[float, List[float]]:
        """Alias for dep_lambda."""
        return self.dep_lambda
    
    @property
    def damping(self) -> Union[float, List[float]]:
        """Alias for p0."""
        return self.p0
    
    @property
    def epsilon_x(self) -> Union[float, List[float]]:
        """Backward compatibility alias for coherent_x."""
        return self.coherent_x
    
    @property
    def epsilon_z(self) -> Union[float, List[float]]:
        """Backward compatibility alias for coherent_y."""
        return self.coherent_y
    
    def get_depolarizing_list(self, qubits: int) -> List[float]:
        """Get depolarizing as list per qubit.
        
        Args:
            qubits: Number of qubits
            
        Returns:
            List of depolarizing parameters
        """
        return self._get_as_list(self.dep_lambda, qubits)
    
    def get_damping_list(self, qubits: int) -> List[float]:
        """Get damping as list per qubit.
        
        Args:
            qubits: Number of qubits
            
        Returns:
            List of damping parameters
        """
        return self._get_as_list(self.p0, qubits)
    
    def get_coherent_x_list(self, qubits: int) -> List[float]:
        """Get coherent_x as list per qubit.
        
        Args:
            qubits: Number of qubits
            
        Returns:
            List of coherent X parameters
        """
        return self._get_as_list(self.coherent_x, qubits)
    
    def get_coherent_y_list(self, qubits: int) -> List[float]:
        """Get coherent_y as list per qubit.
        
        Args:
            qubits: Number of qubits
            
        Returns:
            List of coherent Y parameters
        """
        return self._get_as_list(self.coherent_y, qubits)
    
    def __str__(self) -> str:
        """String representation showing key parameters."""
        # Format noise parameters
        def format_param(val):
            if isinstance(val, list):
                return "[" + ", ".join(f"{v:.4f}" for v in val) + "]"
            return f"{val:.4f}"
        
        # Build noise application info
        noise_apps = []
        if self.x_coherent_on_gate:
            noise_apps.append(f"    • X coherent → {', '.join(self.x_coherent_on_gate)}")
        if self.z_coherent_on_gate:
            noise_apps.append(f"    • Z coherent → {', '.join(self.z_coherent_on_gate)}")
        if self.damping_on_gate:
            noise_apps.append(f"    • Damping    → {', '.join(self.damping_on_gate)}")
        if self.depol_on_gate:
            noise_apps.append(f"    • Depolarize → {', '.join(self.depol_on_gate)}")
        
        noise_app_str = "\n".join(noise_apps) if noise_apps else "    (none configured)"
        
        return (
            f"\n{'='*60}\n"
            f"  NoiseConfig\n"
            f"{'='*60}\n"
            f"  Noise Parameters:\n"
            f"    • Depolarizing:    {format_param(self.dep_lambda)}\n"
            f"    • Damping:         {format_param(self.p0)}\n"
            f"    • Coherent X:      {format_param(self.coherent_x)}\n"
            f"    • Coherent Y:      {format_param(self.coherent_y)}\n"
            f"  \n"
            f"  Noise Application:\n"
            f"{noise_app_str}\n"
            f"{'='*60}"
        )


class DatasetConfig(BaseModel):
    """Configuration for dataset generation.
    
    Attributes:
        n_circuits: Number of circuits to generate
        eval_size: Number of circuits for evaluation dataset
        eval_depth: Circuit depth for evaluation dataset
        moments: Number of moments (circuit depth) for training
        qubits: Number of qubits in circuits
        primitive_gates: List of primitive gate names (e.g., ['rx', 'rz', 'cz'])
        distributed_clifford: Use distributed Clifford gates
        clifford: Generate Clifford circuits (quantized angles)
        mixed: Mix random and Clifford circuits
    """
    
    n_circuits: int = Field(default=100, gt=0)
    eval_size: int = Field(default=100, gt=0)
    eval_depth: int = Field(default=15, gt=0)
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
            f"  Training Dataset:\n"
            f"    • Circuits:       {self.n_circuits}\n"
            f"    • Qubits:         {self.qubits}\n"
            f"    • Moments:        {self.moments}\n"
            f"    • Type:           {circuit_type}\n"
            f"    • Gates:          [{gates_str}]\n"
            f"  \n"
            f"  Evaluation Dataset:\n"
            f"    • Circuits:       {self.eval_size}\n"
            f"    • Moments:        {self.eval_depth}\n"
            f"{'='*50}"
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
