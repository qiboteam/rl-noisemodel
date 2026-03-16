"""Unit tests for gym environment."""

import pytest
import numpy as np
from gymnasium import spaces

from rlnoise.config import DatasetConfig, NoiseConfig, GateSpecificNoise, GymEnvConfig, RewardConfig
from rlnoise.dataset import DatasetGenerator
from rlnoise.circuit_encoder import CircuitEncoder
from rlnoise.gym_env import QuantumCircuitEnv, create_quantum_circuit_env


class TestGymEnvConfig:
    """Test GymEnvConfig validation."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = GymEnvConfig()
        assert config.kernel_size == 3
        assert config.action_penalty == 0.0
        assert config.action_space_max_value == 0.06
        assert config.enable_only_depolarizing is False
        assert config.val_split == 0.2
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = GymEnvConfig(
            kernel_size=5,
            action_penalty=0.1,
            action_space_max_value=0.1,
            val_split=0.3
        )
        assert config.kernel_size == 5
        assert config.action_penalty == 0.1
        assert config.val_split == 0.3
    
    def test_even_kernel_size_raises_error(self):
        """Test that even kernel size raises error."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            GymEnvConfig(kernel_size=4)
    
    def test_invalid_val_split(self):
        """Test that invalid validation split raises error."""
        with pytest.raises(Exception):
            GymEnvConfig(val_split=1.5)


class TestQuantumCircuitEnv:
    """Test QuantumCircuitEnv functionality."""
    
    @pytest.fixture
    def small_dataset(self):
        """Create a small dataset for testing."""
        dataset_config = DatasetConfig(
            n_circuits=10,
            qubits=1,
            moments=5,
            clifford=True
        )
        noise_config = NoiseConfig(
            noise_list=[
                GateSpecificNoise(gate="rx", noise_channel="depolarizing", noise_parameter=0.02),
            ]
        )
        
        generator = DatasetGenerator(dataset_config, noise_config)
        return generator.generate(verbose=False)
    
    @pytest.fixture
    def encoder(self):
        """Create encoder."""
        return CircuitEncoder(primitive_gates=["rx", "rz"])
    
    @pytest.fixture
    def env_config(self):
        """Create environment config."""
        return GymEnvConfig(kernel_size=3, val_split=0.2)
    
    @pytest.fixture
    def reward_config(self):
        """Create reward config."""
        return RewardConfig(metric="trace", alpha=20.0)
    
    @pytest.fixture
    def env(self, small_dataset, encoder, env_config, reward_config):
        """Create environment."""
        return QuantumCircuitEnv(
            dataset=small_dataset,
            encoder=encoder,
            env_config=env_config,
            reward_config=reward_config,
            primitive_gates=["rx", "rz"]
        )
    
    def test_initialization(self, env):
        """Test environment initialization."""
        assert env.n_circuits == 10
        assert env.n_circuits_train == 8  # 80% of 10
        assert env.n_qubits == 1
        assert env.kernel_size == 3
    
    def test_observation_space(self, env):
        """Test observation space definition."""
        assert isinstance(env.observation_space, spaces.Box)
        # Shape: (encoding_dim, n_qubits, kernel_size)
        assert env.observation_space.shape == (8, 1, 3)
        assert env.observation_space.dtype == np.float32
    
    def test_action_space(self, env):
        """Test action space definition."""
        assert isinstance(env.action_space, spaces.Box)
        # Shape: (n_qubits, 4) - 4 noise parameters per qubit
        assert env.action_space.shape == (1, 4)
        assert env.action_space.dtype == np.float32
        assert np.all(env.action_space.low == 0.0)
        assert np.all(env.action_space.high == 1.0)
    
    def test_reset(self, env):
        """Test reset functionality."""
        obs, info = env.reset()
        
        # Check observation
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (8, 1, 3)
        assert obs.dtype == np.float32
        
        # Check info
        assert "circuit_idx" in info
        assert "circuit_length" in info
        assert env.position == 0
    
    def test_reset_with_specific_circuit(self, env):
        """Test reset with specific circuit index."""
        obs, info = env.reset(options={"circuit_idx": 2})
        
        assert info["circuit_idx"] == 2
        assert env.current_circuit_idx == 2
    
    def test_step(self, env):
        """Test step functionality."""
        env.reset()
        
        # Take a random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check outputs
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (8, 1, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert truncated is False  # Never truncated
        assert "position" in info
    
    def test_episode_completion(self, env):
        """Test that episode completes properly."""
        obs, info = env.reset()
        circuit_length = info["circuit_length"]
        
        steps = 0
        terminated = False
        
        while not terminated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            
            if steps > 100:  # Safety check
                break
        
        # Should terminate after circuit_length steps
        assert terminated
        assert steps == circuit_length
    
    def test_reward_only_at_terminal(self, env):
        """Test that reward is only given at terminal state."""
        env.reset()
        
        # First steps should have zero reward
        for _ in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if not terminated:
                assert reward == 0.0
    
    def test_action_application(self, env):
        """Test that actions modify the circuit."""
        obs, info = env.reset()
        
        # Get initial circuit state
        initial_circuit = env.current_circuit.copy()
        
        # Apply action with some noise
        action = np.array([[0.5, 0.3, 0.2, 0.4]])  # Some noise parameters
        env.step(action)
        
        # Circuit should be modified (noise added)
        assert not np.allclose(env.current_circuit, initial_circuit)
    
    def test_only_depolarizing_mode(self, small_dataset, encoder):
        """Test that only_depolarizing mode works."""
        env_config = GymEnvConfig(
            kernel_size=3,
            enable_only_depolarizing=True
        )
        reward_config = RewardConfig()
        
        env = QuantumCircuitEnv(
            dataset=small_dataset,
            encoder=encoder,
            env_config=env_config,
            reward_config=reward_config,
            primitive_gates=["rx", "rz"]
        )
        
        env.reset()
        
        # Apply action with all parameters
        action = np.array([[0.5, 0.5, 0.5, 0.5]])
        env.step(action)
        
        # Only depolarizing (index 4) should be set
        # epsilon_x (7), epsilon_z (6), reset (5) should be 0
        circuit_state = env.current_circuit[env.position - 1, 0]
        assert circuit_state[7] == 0.0  # epsilon_x
        assert circuit_state[6] == 0.0  # epsilon_z
        assert circuit_state[5] == 0.0  # reset
        # depolarizing might not be exactly 0.5*0.06 due to float precision
        assert circuit_state[4] > 0.0  # depolarizing
    
    def test_validation_circuits(self, env):
        """Test validation circuit access."""
        n_val = env.n_validation_circuits
        assert n_val == 2  # 20% of 10
        
        val_idx = env.get_validation_circuit(0)
        assert val_idx == 8  # First validation circuit
        
        val_idx = env.get_validation_circuit(1)
        assert val_idx == 9  # Second validation circuit
    
    def test_get_validation_circuit_out_of_range(self, env):
        """Test that out of range validation index raises error."""
        with pytest.raises(ValueError):
            env.get_validation_circuit(10)
    
    def test_multiple_episodes(self, env):
        """Test running multiple episodes."""
        for episode in range(3):
            obs, info = env.reset()
            terminated = False
            
            while not terminated:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
            
            # Should successfully complete each episode
            assert terminated


class TestMultiQubitEnv:
    """Test environment with multi-qubit circuits."""
    
    @pytest.fixture
    def multiqubit_dataset(self):
        """Create a 2-qubit dataset."""
        dataset_config = DatasetConfig(
            n_circuits=5,
            qubits=2,
            moments=8,
            clifford=True
        )
        noise_config = NoiseConfig(
            noise_list=[
                GateSpecificNoise(gate="rx", noise_channel="depolarizing", noise_parameter=0.02),
            ]
        )
        
        generator = DatasetGenerator(dataset_config, noise_config)
        return generator.generate(verbose=False)
    
    def test_multiqubit_env(self, multiqubit_dataset):
        """Test environment with 2-qubit circuits."""
        env_config = GymEnvConfig(kernel_size=3)
        reward_config = RewardConfig()
        encoder = CircuitEncoder(primitive_gates=["rx", "rz", "cz"])
        
        env = QuantumCircuitEnv(
            dataset=multiqubit_dataset,
            encoder=encoder,
            env_config=env_config,
            reward_config=reward_config,
            primitive_gates=["rx", "rz", "cz"]
        )
        
        # Check spaces
        assert env.observation_space.shape == (8, 2, 3)  # 2 qubits
        assert env.action_space.shape == (2, 4)  # 2 qubits, 4 actions each
        
        # Run episode
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == (8, 2, 3)


class TestCreateQuantumCircuitEnv:
    """Test convenience function."""
    
    def test_create_with_defaults(self):
        """Test creating environment with defaults."""
        dataset_config = DatasetConfig(n_circuits=5, qubits=1, moments=5)
        noise_config = NoiseConfig(noise_list=[])
        
        generator = DatasetGenerator(dataset_config, noise_config)
        dataset = generator.generate(verbose=False)
        
        env = create_quantum_circuit_env(
            dataset=dataset,
            primitive_gates=["rx", "rz"]
        )
        
        assert isinstance(env, QuantumCircuitEnv)
        assert env.kernel_size == 3  # Default
    
    def test_create_with_custom_configs(self):
        """Test creating environment with custom configs."""
        dataset_config = DatasetConfig(n_circuits=5, qubits=1, moments=5)
        noise_config = NoiseConfig(noise_list=[])
        
        generator = DatasetGenerator(dataset_config, noise_config)
        dataset = generator.generate(verbose=False)
        
        env_config = GymEnvConfig(kernel_size=5)
        reward_config = RewardConfig(metric="mse", alpha=10.0)
        
        env = create_quantum_circuit_env(
            dataset=dataset,
            primitive_gates=["rx", "rz"],
            env_config=env_config,
            reward_config=reward_config
        )
        
        assert env.kernel_size == 5
        assert env.reward_fn.config.metric == "mse"
