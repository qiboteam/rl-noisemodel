"""Tests for benchmarking utilities."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing

from rlnoise.benchmarking import (
    maximally_mixed_state,
    generate_rb_circuits,
    fit_rb_decay,
    evaluate_benchmarks,
    _build_composite_circuit,
    _apply_rb_noise_model,
)
from rlnoise.config import DatasetConfig, NoiseConfig, GateSpecificNoise
from rlnoise.circuit_generator import CircuitGenerator
from rlnoise.circuit_encoder import CircuitEncoder
from rlnoise.noise_model import QuantumNoiseModel


class TestMaximallyMixedState:
    """Tests for the maximally_mixed_state helper."""

    def test_qubit_1(self):
        mms = maximally_mixed_state(2)
        assert mms.shape == (2, 2)
        assert np.allclose(mms, np.eye(2) / 2)

    def test_qubit_2(self):
        mms = maximally_mixed_state(4)
        assert mms.shape == (4, 4)
        assert np.allclose(np.trace(mms), 1.0)

    def test_complex_dtype(self):
        mms = maximally_mixed_state(2)
        assert np.iscomplexobj(mms)


class TestBuildCompositeCircuit:
    """Tests for _build_composite_circuit."""

    def test_doubles_circuit_depth(self):
        from qibo.models import Circuit
        from qibo import gates
        circuit = Circuit(1, density_matrix=True)
        circuit.add(gates.RX(0, theta=np.pi / 4))
        composite = _build_composite_circuit(circuit)
        # Composite should have at least twice the gates
        assert len(composite.queue) >= 2

    def test_returns_same_nqubits(self):
        from qibo.models import Circuit
        from qibo import gates
        circuit = Circuit(2, density_matrix=True)
        circuit.add(gates.RX(0, theta=0.1))
        circuit.add(gates.RZ(1, theta=0.2))
        composite = _build_composite_circuit(circuit)
        assert composite.nqubits == 2


class TestApplyRbNoiseModel:
    """Tests for _apply_rb_noise_model."""

    def test_returns_circuit(self):
        from qibo.models import Circuit
        from qibo import gates
        circuit = Circuit(1, density_matrix=True)
        circuit.add(gates.RX(0, theta=0.1))
        noisy = _apply_rb_noise_model(circuit, lambda_rb=0.9)
        assert noisy is not None

    def test_perfect_lambda_no_noise(self):
        from qibo.models import Circuit
        from qibo import gates
        circuit = Circuit(1, density_matrix=True)
        circuit.add(gates.RX(0, theta=0.1))
        # lambda_rb=1 → p=0, no depolarizing noise
        noisy = _apply_rb_noise_model(circuit, lambda_rb=1.0)
        assert noisy is not None


@pytest.fixture
def rb_setup():
    """Create components needed for RB tests."""
    config = DatasetConfig(
        n_circuits=3,
        qubits=1,
        moments=4,
        clifford=True,
        primitive_gates=["rx", "rz"],
    )
    noise_config = NoiseConfig(
        noise_list=[
            GateSpecificNoise(gate="rx", noise_channel="depolarizing", noise_parameter=0.01),
            GateSpecificNoise(gate="rz", noise_channel="depolarizing", noise_parameter=0.01),
        ]
    )
    circuit_gen = CircuitGenerator(config)
    encoder = CircuitEncoder(primitive_gates=["rx", "rz"])
    noise_model = QuantumNoiseModel(noise_config, qubits=1)
    return circuit_gen, encoder, noise_model


class TestGenerateRbCircuits:
    """Tests for generate_rb_circuits."""

    def test_returns_list_per_depth(self, rb_setup):
        circuit_gen, encoder, noise_model = rb_setup
        depths = [3, 5]
        rb_data = generate_rb_circuits(
            circuit_gen, encoder, noise_model,
            depths=depths, n_circuits_per_depth=2
        )
        assert len(rb_data) == len(depths)

    def test_entry_structure(self, rb_setup):
        circuit_gen, encoder, noise_model = rb_setup
        rb_data = generate_rb_circuits(
            circuit_gen, encoder, noise_model,
            depths=[4], n_circuits_per_depth=2
        )
        entry = rb_data[0]
        assert "depth" in entry
        assert "circuit_arrays" in entry
        assert "qibo_circuits" in entry
        assert "labels" in entry
        assert entry["depth"] == 4
        assert len(entry["qibo_circuits"]) == 2

    def test_labels_shape(self, rb_setup):
        circuit_gen, encoder, noise_model = rb_setup
        rb_data = generate_rb_circuits(
            circuit_gen, encoder, noise_model,
            depths=[3], n_circuits_per_depth=2
        )
        labels = rb_data[0]["labels"]
        # 1-qubit → 2x2 density matrices
        assert labels.shape[1:] == (2, 2)

    def test_restores_original_moments(self, rb_setup):
        circuit_gen, encoder, noise_model = rb_setup
        original = circuit_gen.n_moments
        generate_rb_circuits(
            circuit_gen, encoder, noise_model,
            depths=[6], n_circuits_per_depth=1
        )
        assert circuit_gen.n_moments == original


class TestFitRbDecay:
    """Tests for fit_rb_decay."""

    def test_returns_tuple(self, rb_setup):
        circuit_gen, encoder, noise_model = rb_setup
        rb_data = generate_rb_circuits(
            circuit_gen, encoder, noise_model,
            depths=[3, 5, 7], n_circuits_per_depth=2
        )
        a, lam = fit_rb_decay(rb_data, noise_model)
        assert isinstance(a, float)
        assert isinstance(lam, float)

    def test_lambda_in_valid_range(self, rb_setup):
        circuit_gen, encoder, noise_model = rb_setup
        rb_data = generate_rb_circuits(
            circuit_gen, encoder, noise_model,
            depths=[3, 5, 7], n_circuits_per_depth=2
        )
        _, lam = fit_rb_decay(rb_data, noise_model)
        assert 0.0 <= lam <= 1.0

    def test_fallback_on_curve_fit_failure(self, rb_setup):
        """Test that the log-space fallback is used when curve_fit raises RuntimeError."""
        circuit_gen, encoder, noise_model = rb_setup
        rb_data = generate_rb_circuits(
            circuit_gen, encoder, noise_model,
            depths=[3, 5, 7], n_circuits_per_depth=2
        )
        with patch("rlnoise.benchmarking.curve_fit", side_effect=RuntimeError("convergence error")):
            a, lam = fit_rb_decay(rb_data, noise_model)
        assert isinstance(a, float)
        assert isinstance(lam, float)
        assert a == 1.0  # fallback sets a=1.0
        assert 0.0 <= lam <= 1.0


class TestEvaluateBenchmarks:
    """Tests for evaluate_benchmarks."""

    def test_returns_expected_keys(self, rb_setup):
        """evaluate_benchmarks returns a dict with the correct top-level keys."""
        circuit_gen, encoder, noise_model = rb_setup
        rb_data = generate_rb_circuits(
            circuit_gen, encoder, noise_model,
            depths=[3], n_circuits_per_depth=2
        )
        _, lam = fit_rb_decay(rb_data, noise_model)

        from rlnoise.rl_agent import RLAgent
        from rlnoise.config import GymEnvConfig, RewardConfig, AgentConfig
        from rlnoise.dataset import DatasetGenerator
        from rlnoise.gym_env import QuantumCircuitEnv

        dataset_config = DatasetConfig(
            n_circuits=4, qubits=1, moments=3, clifford=True,
            primitive_gates=["rx", "rz"],
        )
        dataset = DatasetGenerator(dataset_config, noise_model.config).generate()
        env = QuantumCircuitEnv(
            dataset=dataset,
            encoder=encoder,
            env_config=GymEnvConfig(kernel_size=3, action_space_max_value=0.1, val_split=0.0),
            reward_config=RewardConfig(metric="trace", function="inverted_squared", alpha=20.0),
        )
        agent = RLAgent(env=env, agent_config=AgentConfig(
            n_steps=100, batch_size=25, features_dim=16, n_filters=8,
        ))

        results = evaluate_benchmarks(rb_data=rb_data, encoder=encoder, rl_agent=agent, lambda_rb=lam)

        assert "depths" in results
        for model_key in ("rl", "rb", "no_noise", "mms"):
            assert model_key in results
            for metric in ("fidelity", "fidelity_std", "trace", "trace_std", "mse", "mse_std"):
                assert metric in results[model_key]

    def test_depths_match_input(self, rb_setup):
        """depths list in results matches the input RB depths."""
        circuit_gen, encoder, noise_model = rb_setup
        depths = [3, 5]
        rb_data = generate_rb_circuits(
            circuit_gen, encoder, noise_model,
            depths=depths, n_circuits_per_depth=2
        )
        _, lam = fit_rb_decay(rb_data, noise_model)

        from rlnoise.rl_agent import RLAgent
        from rlnoise.config import GymEnvConfig, RewardConfig, AgentConfig
        from rlnoise.dataset import DatasetGenerator
        from rlnoise.gym_env import QuantumCircuitEnv

        dataset_config = DatasetConfig(
            n_circuits=4, qubits=1, moments=3, clifford=True,
            primitive_gates=["rx", "rz"],
        )
        dataset = DatasetGenerator(dataset_config, noise_model.config).generate()
        env = QuantumCircuitEnv(
            dataset=dataset,
            encoder=encoder,
            env_config=GymEnvConfig(kernel_size=3, action_space_max_value=0.1, val_split=0.0),
            reward_config=RewardConfig(metric="trace", function="inverted_squared", alpha=20.0),
        )
        agent = RLAgent(env=env, agent_config=AgentConfig(
            n_steps=100, batch_size=25, features_dim=16, n_filters=8,
        ))

        results = evaluate_benchmarks(rb_data=rb_data, encoder=encoder, rl_agent=agent, lambda_rb=lam)

        assert results["depths"] == depths
        assert len(results["rl"]["fidelity"]) == len(depths)

    def test_metric_values_in_range(self, rb_setup):
        """Fidelity is in [0,1] and trace/mse are non-negative."""
        circuit_gen, encoder, noise_model = rb_setup
        rb_data = generate_rb_circuits(
            circuit_gen, encoder, noise_model,
            depths=[3], n_circuits_per_depth=2
        )
        _, lam = fit_rb_decay(rb_data, noise_model)

        from rlnoise.rl_agent import RLAgent
        from rlnoise.config import GymEnvConfig, RewardConfig, AgentConfig
        from rlnoise.dataset import DatasetGenerator
        from rlnoise.gym_env import QuantumCircuitEnv

        dataset_config = DatasetConfig(
            n_circuits=4, qubits=1, moments=3, clifford=True,
            primitive_gates=["rx", "rz"],
        )
        dataset = DatasetGenerator(dataset_config, noise_model.config).generate()
        env = QuantumCircuitEnv(
            dataset=dataset,
            encoder=encoder,
            env_config=GymEnvConfig(kernel_size=3, action_space_max_value=0.1, val_split=0.0),
            reward_config=RewardConfig(metric="trace", function="inverted_squared", alpha=20.0),
        )
        agent = RLAgent(env=env, agent_config=AgentConfig(
            n_steps=100, batch_size=25, features_dim=16, n_filters=8,
        ))

        results = evaluate_benchmarks(rb_data=rb_data, encoder=encoder, rl_agent=agent, lambda_rb=lam)

        for model_key in ("rl", "rb", "no_noise", "mms"):
            for f in results[model_key]["fidelity"]:
                assert -0.01 <= f <= 1.01, f"fidelity {f} out of range for {model_key}"
            for t in results[model_key]["trace"]:
                assert t >= -0.01, f"trace distance {t} negative for {model_key}"
            for m in results[model_key]["mse"]:
                assert m >= -1e-9, f"mse {m} negative for {model_key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

