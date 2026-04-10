"""Tests for training callbacks."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

from rlnoise.callback import TrainingCallback
from rlnoise.config import DatasetConfig, NoiseConfig, GymEnvConfig, RewardConfig, GateSpecificNoise
from rlnoise.dataset import DatasetGenerator
from rlnoise.circuit_encoder import CircuitEncoder
from rlnoise.gym_env import QuantumCircuitEnv


class TestTrainingCallback:
    """Test suite for training callback."""
    
    @pytest.fixture
    def simple_env(self):
        """Create a simple environment for testing."""
        # Create minimal dataset
        dataset_config = DatasetConfig(
            n_circuits=5,
            moments=4,
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
            val_split=0.2
        )
        
        reward_config = RewardConfig(metric="trace", function="inverted_squared", alpha=20.0)
        
        env = QuantumCircuitEnv(dataset, encoder, env_config, reward_config)
        return env
    
    def test_initialization(self, simple_env):
        """Test callback initialization."""
        callback = TrainingCallback(
            env=simple_env,
            check_freq=100,
            save_path="test_model",
            save_best=True,
            verbose=1
        )
        
        assert callback.env == simple_env
        assert callback.check_freq == 100
        assert callback.save_path == "test_model"
        assert callback.save_best is True
        assert callback.best_mean_reward == -np.inf
        assert len(callback.eval_results) == 0
        assert len(callback.train_results) == 0
        assert len(callback.timestep_list) == 0
    
    def test_initialization_without_save(self, simple_env):
        """Test callback initialization without saving."""
        callback = TrainingCallback(
            env=simple_env,
            check_freq=50,
            save_best=False,
            verbose=0
        )
        
        assert callback.save_path is None
        assert callback.save_best is False
    
    def test_on_step_not_at_check_freq(self, simple_env):
        """Test that _on_step returns True when not at check frequency."""
        callback = TrainingCallback(env=simple_env, check_freq=100)
        
        # Mock the parent class attributes
        callback.n_calls = 50
        callback.num_timesteps = 50
        
        # Should return True and not evaluate
        result = callback._on_step()
        assert result is True
        assert len(callback.timestep_list) == 0
    
    def test_evaluate_on_set(self, simple_env):
        """Test evaluation on a dataset."""
        callback = TrainingCallback(env=simple_env, check_freq=100)
        
        # Mock the model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=(np.zeros((1, 4)), None))
        callback.model = mock_model
        
        # Evaluate on training set
        metrics = callback._evaluate_on_set(train=True)
        
        assert isinstance(metrics, np.ndarray)
        assert metrics.shape == (6,)  # mean_reward, std_reward, mean_trace, std_trace, mean_fidelity, std_fidelity
        assert not np.isnan(metrics).any()
        assert np.isfinite(metrics).all()
    
    def test_evaluate_splits_datasets_correctly(self, simple_env):
        """Test that evaluation correctly splits train/val sets."""
        callback = TrainingCallback(env=simple_env, check_freq=100)
        
        # Mock the model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=(np.zeros((1, 4)), None))
        callback.model = mock_model
        
        # Get train and val counts
        n_train = simple_env.n_circuits_train
        n_val = simple_env.n_circuits - n_train
        
        # Evaluate train set
        _ = callback._evaluate_on_set(train=True)
        train_calls = mock_model.predict.call_count
        
        # Reset mock
        mock_model.reset_mock()
        
        # Evaluate val set (if exists)
        if n_val > 0:
            _ = callback._evaluate_on_set(train=False)
            val_calls = mock_model.predict.call_count
            
            # Each circuit requires circuit_length steps
            # So calls should be proportional to number of circuits
            assert train_calls > 0
            assert val_calls > 0
    
    def test_get_results(self, simple_env):
        """Test getting results dictionary."""
        callback = TrainingCallback(env=simple_env, check_freq=100)
        
        # Add some mock results
        callback.timestep_list = [100, 200, 300]
        callback.train_results = [
            np.array([1.0, 0.1]),
            np.array([1.5, 0.2]),
            np.array([2.0, 0.15])
        ]
        callback.eval_results = [
            np.array([0.9, 0.15]),
            np.array([1.2, 0.18]),
            np.array([1.8, 0.12])
        ]
        callback.best_mean_reward = 1.8
        
        results = callback.get_results()
        
        assert "timesteps" in results
        assert "train_results" in results
        assert "eval_results" in results
        assert "best_mean_reward" in results
        assert results["timesteps"] == [100, 200, 300]
        assert len(results["train_results"]) == 3
        assert len(results["eval_results"]) == 3
        assert results["best_mean_reward"] == 1.8
    
    def test_print_metrics(self, simple_env, capsys):
        """Test metrics printing."""
        callback = TrainingCallback(env=simple_env, check_freq=100, verbose=1)
        
        metrics = np.array([1.234, 0.567])
        callback._print_metrics("Test Set", metrics)
        
        captured = capsys.readouterr()
        assert "reward" in captured.out
        assert "1.2340" in captured.out
        assert "0.5670" in captured.out
    
    def test_save_path_directory_creation(self, simple_env):
        """Test that callback creates save directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "subfolder" / "model"
            
            callback = TrainingCallback(
                env=simple_env,
                check_freq=100,
                save_path=str(save_path)
            )
            
            assert save_path.parent.exists()
    
    def test_callback_with_no_validation_set(self):
        """Test callback with environment that has no validation split."""
        # Create dataset with no validation split
        dataset_config = DatasetConfig(
            n_circuits=3,
            moments=3,
            qubits=1,
            primitive_gates=["rx"],
            clifford=True
        )
        
        noise_config = NoiseConfig(
            noise_list=[
                GateSpecificNoise(gate="rx", noise_channel="depolarizing", noise_parameter=0.01),
            ]
        )
        
        generator = DatasetGenerator(dataset_config, noise_config)
        dataset = generator.generate(verbose=False)
        
        encoder = CircuitEncoder(primitive_gates=["rx"])
        
        env_config = GymEnvConfig(
            kernel_size=3,
            val_split=0.0  # No validation split
        )
        
        reward_config = RewardConfig()
        
        env = QuantumCircuitEnv(dataset, encoder, env_config, reward_config)
        
        callback = TrainingCallback(env=env, check_freq=100)
        
        # Mock model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=(np.zeros((1, 4)), None))
        callback.model = mock_model
        callback.num_timesteps = 100
        callback.n_calls = 100
        
        # Evaluate - should not crash with no validation set
        callback._evaluate()
        
        assert len(callback.train_results) == 1
        assert len(callback.eval_results) == 1  # zeros appended when no val set
        assert np.all(callback.eval_results[0] == 0.0)


    def test_save_and_load_history(self, simple_env, tmp_path):
        """Test save_history and load_history round-trip."""
        callback = TrainingCallback(env=simple_env, check_freq=50)

        callback.timestep_list = [50, 100]
        callback.train_results = [
            np.array([1.0, 0.1, 0.5, 0.05]),
            np.array([1.2, 0.2, 0.6, 0.07]),
        ]
        callback.eval_results = [
            np.array([0.9, 0.15, 0.45, 0.06]),
            np.array([1.1, 0.18, 0.55, 0.08]),
        ]
        callback.best_mean_reward = 1.1

        filepath = str(tmp_path / "hist")
        callback.save_history(filepath)

        assert (tmp_path / "hist.npz").exists()

        loaded = TrainingCallback.load_history(filepath)

        assert loaded["timesteps"] == [50, 100]
        assert abs(loaded["best_mean_reward"] - 1.1) < 1e-6
        assert loaded["metric_name"] == "trace"
        assert loaded["check_freq"] == 50
        assert loaded["n_qubits"] == simple_env.n_qubits
        assert loaded["n_circuits_train"] == simple_env.n_circuits_train
        assert loaded["n_circuits_val"] == simple_env.n_circuits - simple_env.n_circuits_train

    def test_load_history_adds_npz_extension(self, simple_env, tmp_path):
        """Test that load_history works with or without .npz extension."""
        callback = TrainingCallback(env=simple_env, check_freq=100)
        callback.timestep_list = [100]
        callback.train_results = [np.array([1.0, 0.0, 0.0, 0.0])]
        callback.eval_results = [np.array([1.0, 0.0, 0.0, 0.0])]

        filepath = str(tmp_path / "hist2")
        callback.save_history(filepath)

        # Load with explicit extension
        loaded_ext = TrainingCallback.load_history(filepath + ".npz")
        loaded_no_ext = TrainingCallback.load_history(filepath)
        assert loaded_ext["n_qubits"] == loaded_no_ext["n_qubits"]

    def test_get_results_all_fields(self, simple_env):
        """Test that get_results returns all expected metadata fields."""
        callback = TrainingCallback(env=simple_env, check_freq=50)

        results = callback.get_results()

        for key in ("timesteps", "train_results", "eval_results",
                    "best_mean_reward", "metric_name",
                    "check_freq", "n_qubits", "n_circuits_train", "n_circuits_val"):
            assert key in results, f"Missing key: {key}"

        assert results["check_freq"] == 50
        assert results["n_qubits"] == simple_env.n_qubits
        assert results["n_circuits_train"] == simple_env.n_circuits_train
        assert results["n_circuits_val"] == simple_env.n_circuits - simple_env.n_circuits_train

    def test_non_deterministic_on_step_accumulates_rewards(self, simple_env):
        """Test _on_step accumulates terminal rewards in non-deterministic mode."""
        callback = TrainingCallback(
            env=simple_env, check_freq=1000, deterministic_train_eval=False
        )

        callback.n_calls = 5
        callback.num_timesteps = 5
        callback.locals = {
            "dones": [True, False, True],
            "rewards": [0.8, 0.3, 0.6],
            "infos": [
                {"trace_distance": 0.5, "fidelity": 0.95},
                {},
                {"trace_distance": 0.9, "fidelity": 0.80},
            ],
        }

        result = callback._on_step()
        assert result is True
        assert len(callback._rollout_rewards) == 2
        assert callback._rollout_rewards[0] == pytest.approx(0.8)
        assert callback._rollout_rewards[1] == pytest.approx(0.6)
        assert len(callback._rollout_trace_values) == 2
        assert len(callback._rollout_fidelity_values) == 2

    def test_non_deterministic_evaluate_uses_rollout(self, simple_env):
        """Test _evaluate in non-deterministic mode uses accumulated rollout rewards."""
        callback = TrainingCallback(
            env=simple_env, check_freq=100, deterministic_train_eval=False
        )

        mock_model = Mock()
        mock_model.predict = Mock(return_value=(np.zeros((1, 4)), None))
        callback.model = mock_model
        callback.num_timesteps = 100

        callback._rollout_rewards = [0.5, 0.8, 0.9]
        callback._rollout_trace_values = [0.4, 0.7, 0.8]
        callback._rollout_fidelity_values = [0.6, 0.3, 0.2]

        callback._evaluate()

        assert len(callback.train_results) == 1
        assert callback._rollout_rewards == []   # flushed
        expected_mean = np.mean([0.5, 0.8, 0.9])
        assert callback.train_results[0][0] == pytest.approx(expected_mean)

    def test_non_deterministic_evaluate_empty_rollout(self, simple_env):
        """Test _evaluate in non-deterministic mode with empty rollout uses zeros."""
        callback = TrainingCallback(
            env=simple_env, check_freq=100, deterministic_train_eval=False
        )

        mock_model = Mock()
        mock_model.predict = Mock(return_value=(np.zeros((1, 4)), None))
        callback.model = mock_model
        callback.num_timesteps = 100

        # Empty rollout → should produce zero train_metrics
        callback._rollout_rewards = []

        callback._evaluate()

        assert len(callback.train_results) == 1
        np.testing.assert_array_equal(callback.train_results[0], np.zeros(6))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
