"""Gymnasium environment for quantum circuit noise modeling."""

import random
from typing import Optional, Tuple
import numpy as np
import gymnasium
from gymnasium import spaces

from rlnoise.config import GymEnvConfig, RewardConfig
from rlnoise.dataset import CircuitDataset
from rlnoise.circuit_encoder import CircuitEncoder
from rlnoise.reward import RewardFunction


class QuantumCircuitEnv(gymnasium.Env):  # pylint: disable=too-many-instance-attributes
    """Gymnasium environment for learning quantum noise models.

    The agent observes a sliding window of the circuit and applies noise
    actions (depolarizing, damping, coherent errors) at each position.
    The goal is to match the actual noisy circuit behavior.

    **Observation Space:**
    - Box(encoding_dim, n_qubits, kernel_size) - Sliding window view of circuit

    **Action Space:**
    - Box(n_qubits, 4) - Noise parameters for each qubit:
        - [0]: epsilon_x (coherent X error)
        - [1]: epsilon_z (coherent Z error)
        - [2]: reset/damping probability
        - [3]: depolarizing lambda

    **Reward:**
    - Computed only at terminal state
    - Based on fidelity between predicted and target density matrices

    Args:
        dataset: CircuitDataset with circuits and labels
        encoder: CircuitEncoder for circuit representation
        env_config: GymEnvConfig for environment parameters
        reward_config: RewardConfig for reward function

    Example:
        >>> dataset = CircuitDataset.load("training_data.npz")
        >>> encoder = CircuitEncoder(primitive_gates=["rx", "rz"])
        >>> env_config = GymEnvConfig(kernel_size=3)
        >>> reward_config = RewardConfig(metric="trace", alpha=20.0)
        >>> env = QuantumCircuitEnv(dataset, encoder, env_config, reward_config)
        >>> obs, info = env.reset()
        >>> action = env.action_space.sample()
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset: CircuitDataset,
        encoder: CircuitEncoder,
        env_config: GymEnvConfig,
        reward_config: RewardConfig,
    ):
        super().__init__()

        # Store configuration
        self.dataset = dataset
        self.encoder = encoder
        self.env_config = env_config
        self.reward_config = reward_config

        # Initialize reward function
        self.reward_fn = RewardFunction(reward_config)

        # Dataset properties
        self.n_circuits = len(dataset)
        self.n_circuits_train = int((1 - env_config.val_split) * self.n_circuits)

        # Get circuit properties from first circuit
        example_circuit = dataset.circuits[0]
        self.n_qubits = example_circuit.shape[1]
        self.encoding_dim = example_circuit.shape[2]

        # Environment parameters
        self.kernel_size = env_config.kernel_size
        self.action_max = env_config.action_space_max_value
        self.only_depol = env_config.enable_only_depolarizing

        # Validate kernel size
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        # Define observation space (sliding window)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.encoding_dim, self.n_qubits, self.kernel_size),
            dtype=np.float32,
        )

        # Define action space (noise parameters for each qubit)
        # [epsilon_x, epsilon_z, reset_prob, depol_lambda] for each qubit
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_qubits, 4),
            dtype=np.float32,
        )

        # State variables (set in reset())
        self.current_circuit_idx = None
        self.current_circuit = None
        self.target_dm = None
        self.position = None
        self.circuit_length = None
        self.padded_circuit = None

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"QuantumCircuitEnv("
            f"n_circuits={self.n_circuits}, "
            f"n_qubits={self.n_qubits}, "
            f"kernel_size={self.kernel_size}, "
            f"metric='{self.reward_config.metric}')"
        )

    def __str__(self) -> str:
        """Human-readable representation."""
        lines = [
            "QuantumCircuitEnv:",
            "  Dataset:",
            f"    circuits: {self.n_circuits} ({self.n_circuits_train} train, "
            f"{self.n_circuits - self.n_circuits_train} val)",
            f"    qubits: {self.n_qubits}",
            "  Encoder:",
            f"    primitive_gates: {self.encoder.primitive_gates}",
            "  Environment:",
            f"    observation_space: {self.observation_space.shape}",
            f"    action_space: {self.action_space.shape}",
            f"    kernel_size: {self.kernel_size}",
            f"    action_max: {self.action_max}",
            f"    only_depolarizing: {self.only_depol}",
            "  Reward:",
            f"    metric: {self.reward_config.metric}",
            f"    function: {self.reward_config.function}",
            f"    alpha: {self.reward_config.alpha}",
        ]
        return "\n".join(lines)

    def _pad_circuit(self, circuit: np.ndarray) -> np.ndarray:
        """Add padding to circuit for sliding window.

        Args:
            circuit: Circuit array of shape (n_moments, n_qubits, encoding_dim)

        Returns:
            Padded circuit array
        """
        # Transpose to (encoding_dim, n_qubits, n_moments)
        circuit_t = circuit.transpose(2, 1, 0)

        # Add padding on the time axis
        pad_size = self.kernel_size // 2
        padding = np.zeros(
            (self.encoding_dim, self.n_qubits, pad_size),
            dtype=np.float32
        )

        padded = np.concatenate([padding, circuit_t, padding], axis=2)
        return padded

    def _get_observation(self) -> np.ndarray:
        """Get current observation (sliding window).

        Returns:
            Observation array
        """
        # Update padded circuit with current state
        pad_size = self.kernel_size // 2
        self.padded_circuit[:, :, pad_size:-pad_size] = self.current_circuit.transpose(2, 1, 0)

        # Extract window at current position
        window = self.padded_circuit[
            :, :, self.position : self.position + self.kernel_size
        ]

        return window.astype(np.float32)

    def _apply_action(self, action: np.ndarray):
        """Apply noise action to current circuit position.

        Args:
            action: Action array of shape (n_qubits, 4)
        """
        # Scale action by maximum value
        scaled_action = action * self.action_max

        # If only depolarizing, zero out other actions
        if self.only_depol:
            scaled_action[:, :3] = 0.0

        # Apply action to circuit at current position
        # Action indices: 0=epsilon_x, 1=epsilon_z, 2=reset, 3=depol
        for qubit in range(self.n_qubits):
            # epsilon_x (index 0 -> encoding index 7)
            self.current_circuit[self.position, qubit, 7] = scaled_action[qubit, 0]

            # epsilon_z (index 1 -> encoding index 6)
            self.current_circuit[self.position, qubit, 6] = scaled_action[qubit, 1]

            # reset channel (index 2 -> encoding index 5)
            self.current_circuit[self.position, qubit, 5] = scaled_action[qubit, 2]

            # depolarizing (index 3 -> encoding index 4)
            self.current_circuit[self.position, qubit, 4] = scaled_action[qubit, 3]

    def _get_current_circuit(self):
        """Get current circuit as Qibo circuit object."""
        return self.encoder.array_to_circuit(self.current_circuit)

    def _compute_reward(self, is_terminal: bool) -> float:
        """Compute reward for current state.

        Args:
            is_terminal: Whether this is the terminal state

        Returns:
            Reward value
        """
        if not is_terminal:
            return 0.0

        # Get density matrix from current circuit
        circuit = self._get_current_circuit()
        predicted_dm = circuit().state()

        # Compute reward
        reward = self.reward_fn(predicted_dm, self.target_dm, is_terminal=True)

        return reward

    def reset(  # pylint: disable=arguments-differ
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state.

        Args:
            seed: Random seed
            options: Optional dict with 'circuit_idx' to select specific circuit

        Returns:
            (observation, info)
        """
        super().reset(seed=seed)

        # Select circuit
        if options is not None and "circuit_idx" in options:
            self.current_circuit_idx = options["circuit_idx"]
        else:
            # Random training circuit
            self.current_circuit_idx = random.randint(0, self.n_circuits_train - 1)

        # Load circuit and target
        self.current_circuit = self.dataset.circuits[self.current_circuit_idx].copy()
        self.target_dm = self.dataset.labels[self.current_circuit_idx]

        # Initialize state
        self.circuit_length = self.current_circuit.shape[0]
        self.position = 0

        # Create padded version for sliding window
        self.padded_circuit = self._pad_circuit(self.current_circuit)

        # Get initial observation
        obs = self._get_observation()

        info = {
            "circuit_idx": self.current_circuit_idx,
            "circuit_length": self.circuit_length,
        }

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment.

        Args:
            action: Action array of shape (n_qubits, 4)

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Apply action at current position
        self._apply_action(action)

        # Check if we're at the end
        terminated = self.position >= self.circuit_length - 1

        # Compute reward and raw metric only at terminal state
        reward = 0.0
        if terminated:
            circuit = self._get_current_circuit()
            predicted_dm = circuit().state()
            metric_value = float(self.reward_fn.metric(predicted_dm, self.target_dm))
            reward = float(self.reward_fn.transform(metric_value))

        # Move to next position if not terminated
        if not terminated:
            self.position += 1

        # Get next observation
        obs = self._get_observation()

        # No truncation in this environment
        truncated = False

        info = {
            "position": self.position,
            "circuit_length": self.circuit_length,
        }
        if terminated:
            info["metric"] = metric_value

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render environment (not implemented)."""

    def get_validation_circuit(self, val_idx: int = 0) -> int:
        """Get index of a validation circuit.

        Args:
            val_idx: Index within validation set

        Returns:
            Circuit index in full dataset
        """
        if val_idx >= self.n_circuits - self.n_circuits_train:
            raise ValueError(f"Validation index {val_idx} out of range")

        return self.n_circuits_train + val_idx

    @property
    def n_validation_circuits(self) -> int:
        """Number of validation circuits."""
        return self.n_circuits - self.n_circuits_train


def create_quantum_circuit_env(
    dataset: CircuitDataset,
    primitive_gates: list,
    env_config: Optional[GymEnvConfig] = None,
    reward_config: Optional[RewardConfig] = None,
) -> QuantumCircuitEnv:
    """Convenience factory to create a QuantumCircuitEnv with a new encoder.

    Args:
        dataset: CircuitDataset to use.
        primitive_gates: Gate names for the CircuitEncoder (e.g. ['rx', 'rz']).
        env_config: Optional GymEnvConfig; defaults to GymEnvConfig() if None.
        reward_config: Optional RewardConfig; defaults to RewardConfig() if None.

    Returns:
        Configured QuantumCircuitEnv.
    """
    encoder = CircuitEncoder(primitive_gates=primitive_gates)
    if env_config is None:
        env_config = GymEnvConfig()
    if reward_config is None:
        reward_config = RewardConfig()
    return QuantumCircuitEnv(dataset, encoder, env_config, reward_config)
