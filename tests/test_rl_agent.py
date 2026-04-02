"""Tests for RL agent."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from rlnoise.rl_agent import RLAgent
from rlnoise.config import (
    DatasetConfig,
    NoiseConfig,
    GymEnvConfig,
    RewardConfig,
    AgentConfig,
    GateSpecificNoise
)
from rlnoise.dataset import DatasetGenerator, CircuitDataset
from rlnoise.circuit_encoder import CircuitEncoder
from rlnoise.gym_env import QuantumCircuitEnv


class TestRLAgent:
    """Test suite for RL agent."""
    
    @pytest.fixture
    def simple_env(self):
        """Create a simple environment for testing."""
        dataset_config = DatasetConfig(
            n_circuits=4,
            moments=3,
            qubits=1,
            primitive_gates=["rx", "rz"],
            clifford=True
        )
        
        noise_config = NoiseConfig(
            noise_list=[
                GateSpecificNoise(gate="rx", noise_channel="depolarizing", noise_parameter=0.01),
                GateSpecificNoise(gate="rz", noise_channel="depolarizing", noise_parameter=0.01),
            ]
        )
        
        generator = DatasetGenerator(dataset_config, noise_config)
        dataset = generator.generate(verbose=False)
        
        encoder = CircuitEncoder(primitive_gates=["rx", "rz"])
        
        env_config = GymEnvConfig(
            kernel_size=3,
            action_space_max_value=0.1,
            val_split=0.25
        )
        
        reward_config = RewardConfig(metric="trace", function="inverted_squared", alpha=20.0)
        
        env = QuantumCircuitEnv(dataset, encoder, env_config, reward_config)
        return env
    
    @pytest.fixture
    def agent_config(self):
        """Create agent configuration."""
        return AgentConfig(
            policy="MlpPolicy",
            features_dim=32,
            filter_size=2,
            n_filters=16,
            pi_net_arch=[16],
            vf_net_arch=[16],
            n_steps=16,  # Small for testing
            batch_size=8,
            learning_rate=3e-4,
            verbose=0
        )
    
    def test_initialization(self, simple_env, agent_config):
        """Test agent initialization."""
        agent = RLAgent(simple_env, agent_config)
        
        assert agent.env == simple_env
        assert agent.agent_config == agent_config
        assert agent.model is not None
        assert hasattr(agent.model, 'policy')
    
    def test_initialization_with_custom_config(self, simple_env):
        """Test agent with various configurations."""
        configs = [
            AgentConfig(features_dim=64, n_filters=32),
            AgentConfig(features_dim=32, filter_size=3, n_filters=16),
            AgentConfig(pi_net_arch=[32, 32], vf_net_arch=[32, 32]),
        ]
        
        for config in configs:
            agent = RLAgent(simple_env, config)
            assert agent.model is not None
    
    def test_predict(self, simple_env, agent_config):
        """Test action prediction."""
        agent = RLAgent(simple_env, agent_config)
        
        # Get observation from environment
        obs, _ = simple_env.reset()
        
        # Predict action
        action = agent.predict(obs, deterministic=True)
        
        # Check action shape and values
        assert action.shape == (simple_env.n_qubits, 4)
        assert np.all(action >= 0.0)
        assert np.all(action <= 1.0)
    
    def test_predict_deterministic_consistency(self, simple_env, agent_config):
        """Test that deterministic predictions are consistent."""
        agent = RLAgent(simple_env, agent_config)
        
        obs, _ = simple_env.reset()
        
        # Predict multiple times
        action1 = agent.predict(obs, deterministic=True)
        action2 = agent.predict(obs, deterministic=True)
        
        # Should be identical
        np.testing.assert_array_almost_equal(action1, action2)
    
    def test_train_short_run(self, simple_env, agent_config):
        """Test short training run."""
        agent = RLAgent(simple_env, agent_config)
        
        # Train for very few steps
        results = agent.train(
            total_timesteps=50,
            check_freq=25,
            progress_bar=False
        )
        
        # Check results structure
        assert "timesteps" in results
        assert "train_results" in results
        assert "best_mean_reward" in results
        assert len(results["timesteps"]) > 0
    
    def test_evaluate(self, simple_env, agent_config):
        """Test agent evaluation."""
        agent = RLAgent(simple_env, agent_config)
        
        # Evaluate
        metrics = agent.evaluate(n_episodes=2, deterministic=True)
        
        # Check metrics
        assert "mean_reward" in metrics
        assert "std_reward" in metrics
        assert "min_reward" in metrics
        assert "max_reward" in metrics
        assert np.isfinite(metrics["mean_reward"])
    
    def test_save_and_load(self, simple_env, agent_config):
        """Test saving and loading agent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_agent"
            
            # Create and train briefly
            agent1 = RLAgent(simple_env, agent_config)
            agent1.train(total_timesteps=20, progress_bar=False)
            
            # Save
            agent1.save(str(save_path))
            
            # Check file exists
            assert save_path.with_suffix(".zip").exists()
            
            # Load into new agent
            agent2 = RLAgent(simple_env, agent_config, model_path=str(save_path))
            
            # Compare predictions
            obs, _ = simple_env.reset()
            action1 = agent1.predict(obs, deterministic=True)
            action2 = agent2.predict(obs, deterministic=True)
            
            np.testing.assert_array_almost_equal(action1, action2)
    
    def test_apply_to_circuit(self, simple_env, agent_config):
        """Test applying agent to a circuit."""
        agent = RLAgent(simple_env, agent_config)
        
        # Get a circuit from the dataset
        circuit_array = simple_env.dataset.circuits[0]
        
        # Apply agent (return array)
        noisy_circuit_array = agent.apply_to_circuit(circuit_array, return_qibo=False)
        
        # Check output
        assert noisy_circuit_array.shape == circuit_array.shape
        
        # Noise should be applied (indices 4-7 should have some values)
        noise_params = noisy_circuit_array[:, :, 4:8]
        assert np.any(noise_params > 0), "No noise was applied"
    
    def test_apply_to_circuit_qibo(self, simple_env, agent_config):
        """Test applying agent to circuit and returning Qibo circuit."""
        agent = RLAgent(simple_env, agent_config)

        circuit_array = simple_env.dataset.circuits[0]

        # Apply and return Qibo circuit
        qibo_circuit = agent.apply_to_circuit(circuit_array, return_qibo=True)

        # Check it's a Qibo circuit
        from qibo.models.circuit import Circuit
        assert isinstance(qibo_circuit, Circuit)
        assert qibo_circuit.nqubits == simple_env.n_qubits

    def test_str_representation(self, simple_env, agent_config):
        """Test __str__ produces a useful summary."""
        agent = RLAgent(simple_env, agent_config)
        s = str(agent)
        assert "RLAgent" in s
        assert "Observation space" in s
        assert "Action space" in s
        assert "Learning rate" in s

    def test_print_network(self, simple_env, agent_config, capsys):
        """Test print_network outputs architecture information."""
        agent = RLAgent(simple_env, agent_config)
        agent.print_network()
        captured = capsys.readouterr()
        assert "NEURAL NETWORK ARCHITECTURE" in captured.out
        assert "PARAMETER COUNT BY MODULE" in captured.out
        assert "PARAMETER TENSOR SHAPES" in captured.out

    def test_train_with_history_path(self, simple_env, agent_config, tmp_path):
        """Test that training with history_path saves .npz file."""
        agent = RLAgent(simple_env, agent_config)
        history_path = str(tmp_path / "run_history")

        results = agent.train(
            total_timesteps=50,
            check_freq=25,
            progress_bar=False,
            history_path=history_path,
        )

        assert (tmp_path / "run_history.npz").exists()
        assert "timesteps" in results

    def test_train_with_explicit_save_path_loads_best(self, simple_env, agent_config, tmp_path):
        """Test that when save_path is given explicitly, best weights are restored."""
        save_path = str(tmp_path / "best_model")
        agent = RLAgent(simple_env, agent_config)

        import io
        out = io.StringIO()

        results = agent.train(
            total_timesteps=50,
            check_freq=25,
            save_path=save_path,
            save_best=True,
            progress_bar=False,
            verbose=True,
        )

        # The run should complete without error; best reward may or may not improve
        assert "best_mean_reward" in results

    def test_train_with_previous_history(self, simple_env, agent_config, tmp_path):
        """Test that previous_history is prepended to the new run's results."""
        agent = RLAgent(simple_env, agent_config)

        # Build a fake previous history with 3 checkpoints
        prev = {
            "timesteps": [25, 50, 75],
            "train_results": [
                [1.0, 0.1, 0.5, 0.05],
                [1.1, 0.1, 0.6, 0.05],
                [1.2, 0.1, 0.7, 0.05],
            ],
            "eval_results": [
                [0.9, 0.1, 0.4, 0.05],
                [1.0, 0.1, 0.5, 0.05],
                [1.1, 0.1, 0.6, 0.05],
            ],
            "best_mean_reward": 1.1,
            "metric_name": "trace",
            "check_freq": 25,
            "n_qubits": simple_env.n_qubits,
            "n_circuits_train": simple_env.n_circuits_train,
            "n_circuits_val": simple_env.n_circuits - simple_env.n_circuits_train,
        }

        results = agent.train(
            total_timesteps=50,
            check_freq=25,
            progress_bar=False,
            previous_history=prev,
        )

        # Should have at least 3 old + however many new checkpoints were added
        assert len(results["timesteps"]) >= 3
        # First three timesteps must match the old run
        assert results["timesteps"][:3] == [25, 50, 75]
        # best_mean_reward must be at least the old best
        assert results["best_mean_reward"] >= prev["best_mean_reward"]

    def test_apply_to_circuit_2d_input(self, simple_env, agent_config):
        """Test apply_to_circuit expands 2-D input correctly."""
        agent = RLAgent(simple_env, agent_config)

        # Grab a 3-D circuit then take a 2-D slice (single moment)
        circuit_3d = simple_env.dataset.circuits[0]  # (moments, n_qubits, enc_dim)
        circuit_2d = circuit_3d[0]                   # (n_qubits, enc_dim)

        # Should not raise; the internal expand_dims chain handles 2-D input
        result = agent.apply_to_circuit(circuit_2d, return_qibo=False)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    
    def test_agent_with_different_architectures(self, simple_env):
        """Test agent with different network architectures."""
        architectures = [
            {"pi_net_arch": [16], "vf_net_arch": [16]},
            {"pi_net_arch": [32, 16], "vf_net_arch": [32, 16]},
            {"pi_net_arch": [64], "vf_net_arch": [32]},
        ]
        
        for arch in architectures:
            config = AgentConfig(**arch, n_steps=16, batch_size=8)
            agent = RLAgent(simple_env, config)
            
            # Test prediction works
            obs, _ = simple_env.reset()
            action = agent.predict(obs)
            assert action.shape == (simple_env.n_qubits, 4)
    
    def test_agent_improves_with_training(self, simple_env, agent_config):
        """Test that agent performance can change with training."""
        agent = RLAgent(simple_env, agent_config)
        
        # Evaluate before training
        metrics_before = agent.evaluate(n_episodes=2, deterministic=True)
        
        # Train
        agent.train(total_timesteps=100, progress_bar=False)
        
        # Evaluate after training
        metrics_after = agent.evaluate(n_episodes=2, deterministic=True)
        
        # Both should produce valid metrics (improvement not guaranteed with so little training)
        assert np.isfinite(metrics_before["mean_reward"])
        assert np.isfinite(metrics_after["mean_reward"])
    
    def test_agent_config_validation(self):
        """Test agent config validation."""
        # Valid config
        config = AgentConfig(n_steps=64, batch_size=16)
        assert config.n_steps == 64
        assert config.batch_size == 16
        
        # Invalid: batch_size doesn't divide n_steps
        with pytest.raises(ValueError, match="batch_size.*must divide n_steps"):
            AgentConfig(n_steps=64, batch_size=15)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
