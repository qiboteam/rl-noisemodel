"""Tests for benchmarking utilities."""

import pytest
import numpy as np
from unittest.mock import patch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing

from rlnoise.benchmarking import (
    maximally_mixed_state,
    generate_rb_circuits,
    fit_rb_decay,
    evaluate_benchmarks,
    evaluate_circuit,
    evaluate_on_dataset,
    summarize_benchmarks,
    summarize_rb_parameters,
    summarize_circuit_metrics,
    _build_composite_circuit,
    _apply_rb_noise_model,
)
from rlnoise.config import DatasetConfig, NoiseConfig, GateSpecificNoise
from rlnoise.circuit_generator import CircuitGenerator, grover_circuit, qft_circuit
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


def _make_agent(encoder, noise_model):
    """Helper to build a minimal untrained RLAgent."""
    from rlnoise.rl_agent import RLAgent
    from rlnoise.config import GymEnvConfig, RewardConfig, AgentConfig, DatasetConfig
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
    return RLAgent(env=env, agent_config=AgentConfig(
        n_steps=100, batch_size=25, features_dim=16, n_filters=8,
    ))


class TestEvaluateBenchmarksOptional:
    """Tests for the optional model flags in evaluate_benchmarks."""

    def test_lambda_rb_none_skips_rb(self, rb_setup):
        circuit_gen, encoder, noise_model = rb_setup
        rb_data = generate_rb_circuits(circuit_gen, encoder, noise_model,
                                       depths=[3], n_circuits_per_depth=2)
        agent = _make_agent(encoder, noise_model)
        results = evaluate_benchmarks(rb_data=rb_data, encoder=encoder,
                                      rl_agent=agent, lambda_rb=None)
        assert "rb" not in results
        assert "rl" in results

    def test_evaluate_mms_false(self, rb_setup):
        circuit_gen, encoder, noise_model = rb_setup
        rb_data = generate_rb_circuits(circuit_gen, encoder, noise_model,
                                       depths=[3], n_circuits_per_depth=2)
        agent = _make_agent(encoder, noise_model)
        results = evaluate_benchmarks(rb_data=rb_data, encoder=encoder,
                                      rl_agent=agent, lambda_rb=None,
                                      evaluate_mms=False)
        assert "mms" not in results

    def test_evaluate_no_noise_false(self, rb_setup):
        circuit_gen, encoder, noise_model = rb_setup
        rb_data = generate_rb_circuits(circuit_gen, encoder, noise_model,
                                       depths=[3], n_circuits_per_depth=2)
        agent = _make_agent(encoder, noise_model)
        results = evaluate_benchmarks(rb_data=rb_data, encoder=encoder,
                                      rl_agent=agent, lambda_rb=None,
                                      evaluate_no_noise=False)
        assert "no_noise" not in results

    def test_rl_only_mode(self, rb_setup):
        """All optional models disabled → only rl key present."""
        circuit_gen, encoder, noise_model = rb_setup
        rb_data = generate_rb_circuits(circuit_gen, encoder, noise_model,
                                       depths=[3], n_circuits_per_depth=2)
        agent = _make_agent(encoder, noise_model)
        results = evaluate_benchmarks(rb_data=rb_data, encoder=encoder,
                                      rl_agent=agent, lambda_rb=None,
                                      evaluate_mms=False, evaluate_no_noise=False)
        assert set(results.keys()) == {"depths", "rl"}


class TestSummarizeBenchmarks:
    """Tests for summarize_benchmarks."""

    def test_returns_string(self, rb_setup):
        circuit_gen, encoder, noise_model = rb_setup
        rb_data = generate_rb_circuits(circuit_gen, encoder, noise_model,
                                       depths=[3], n_circuits_per_depth=2)
        agent = _make_agent(encoder, noise_model)
        results = evaluate_benchmarks(rb_data=rb_data, encoder=encoder,
                                      rl_agent=agent, lambda_rb=None,
                                      evaluate_mms=False, evaluate_no_noise=False)
        table = summarize_benchmarks(results)
        assert isinstance(table, str)
        assert "RL model" in table
        assert "Fidelity" in table

    def test_skips_absent_models(self, rb_setup):
        circuit_gen, encoder, noise_model = rb_setup
        rb_data = generate_rb_circuits(circuit_gen, encoder, noise_model,
                                       depths=[3], n_circuits_per_depth=2)
        agent = _make_agent(encoder, noise_model)
        results = evaluate_benchmarks(rb_data=rb_data, encoder=encoder,
                                      rl_agent=agent, lambda_rb=None,
                                      evaluate_mms=False, evaluate_no_noise=False)
        table = summarize_benchmarks(results)
        assert "mms" not in table.lower()
        assert "no noise" not in table.lower()


class TestSummarizeRbParameters:
    """Tests for summarize_rb_parameters."""

    def test_returns_string(self):
        result = summarize_rb_parameters(a=0.95, lambda_rb=0.92)
        assert isinstance(result, str)
        assert "0.95" in result
        assert "0.92" in result

    def test_contains_expected_fields(self):
        result = summarize_rb_parameters(a=1.0, lambda_rb=0.9)
        assert "amplitude" in result.lower() or "a" in result
        assert "decay" in result.lower() or "λ" in result


class TestSaveLoadRbFit:
    """Tests for save_rb_fit and load_rb_fit."""

    def test_round_trip(self, tmp_path):
        from rlnoise.benchmarking import save_rb_fit, load_rb_fit
        path = str(tmp_path / "rb_fit")
        save_rb_fit(0.95, 0.88, path)
        a, lam = load_rb_fit(path)
        assert pytest.approx(a, abs=1e-9) == 0.95
        assert pytest.approx(lam, abs=1e-9) == 0.88

    def test_extension_added_automatically(self, tmp_path):
        from rlnoise.benchmarking import save_rb_fit, load_rb_fit
        path = str(tmp_path / "fit")
        save_rb_fit(1.0, 0.5, path)
        assert (tmp_path / "fit.json").exists()
        a, lam = load_rb_fit(path)
        assert pytest.approx(a) == 1.0

    def test_json_content(self, tmp_path):
        import json
        from rlnoise.benchmarking import save_rb_fit
        path = str(tmp_path / "fit.json")
        save_rb_fit(0.75, 0.62, path)
        with open(path) as f:
            data = json.load(f)
        assert "a" in data and "lambda_rb" in data
        assert pytest.approx(data["a"]) == 0.75
        assert pytest.approx(data["lambda_rb"]) == 0.62

    def test_creates_parent_dirs(self, tmp_path):
        from rlnoise.benchmarking import save_rb_fit
        path = str(tmp_path / "nested" / "dir" / "fit")
        save_rb_fit(0.5, 0.3, path)
        assert (tmp_path / "nested" / "dir" / "fit.json").exists()

    def test_load_missing_file_raises(self, tmp_path):
        from rlnoise.benchmarking import load_rb_fit
        with pytest.raises(FileNotFoundError):
            load_rb_fit(str(tmp_path / "nonexistent.json"))


class TestPlotRbDecay:
    """Tests for plot_rb_decay."""

    def test_returns_figure(self, rb_setup):
        import matplotlib.figure
        from rlnoise.visualization import plot_rb_decay
        circuit_gen, encoder, noise_model = rb_setup
        rb_data = generate_rb_circuits(circuit_gen, encoder, noise_model,
                                       depths=[3, 5], n_circuits_per_depth=2)
        a, lam = fit_rb_decay(rb_data, noise_model)
        fig = plot_rb_decay(rb_data, a=a, lambda_rb=lam)
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_saves_file(self, rb_setup, tmp_path):
        from rlnoise.visualization import plot_rb_decay
        circuit_gen, encoder, noise_model = rb_setup
        rb_data = generate_rb_circuits(circuit_gen, encoder, noise_model,
                                       depths=[3], n_circuits_per_depth=2)
        a, lam = fit_rb_decay(rb_data, noise_model)
        filepath = str(tmp_path / "rb_decay.png")
        fig = plot_rb_decay(rb_data, a=a, lambda_rb=lam, filepath=filepath)
        import matplotlib.pyplot as plt
        plt.close(fig)
        assert (tmp_path / "rb_decay.png").exists()


class TestPlotBenchmarkingResultsUpdated:
    """Tests for updated plot_benchmarking_results with metric flags and missing models."""

    def _minimal_results(self):
        return {
            "depths": [3, 5],
            "rl": {
                "fidelity": [0.8, 0.75], "fidelity_std": [0.01, 0.01],
                "trace":    [0.1, 0.12], "trace_std":    [0.01, 0.01],
                "mse":      [0.01, 0.02], "mse_std":     [0.001, 0.001],
            },
        }

    def test_rl_only_no_error(self):
        from rlnoise.visualization import plot_benchmarking_results
        import matplotlib.pyplot as plt
        results = self._minimal_results()
        fig = plot_benchmarking_results(results, title="RL only")
        assert fig is not None
        plt.close(fig)

    def test_single_metric_panel(self):
        from rlnoise.visualization import plot_benchmarking_results
        import matplotlib.pyplot as plt
        results = self._minimal_results()
        fig = plot_benchmarking_results(results, show_fidelity=True,
                                        show_trace=False, show_mse=False)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_all_metrics_disabled_raises(self):
        from rlnoise.visualization import plot_benchmarking_results
        results = self._minimal_results()
        with pytest.raises(ValueError):
            plot_benchmarking_results(results, show_fidelity=False,
                                      show_trace=False, show_mse=False)

    def test_missing_model_not_plotted(self):
        from rlnoise.visualization import plot_benchmarking_results
        import matplotlib.pyplot as plt
        results = self._minimal_results()  # only "rl" key
        fig = plot_benchmarking_results(results)
        # Should render without error even though rb/no_noise/mms are absent
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# Fixed benchmark circuit builders
# ---------------------------------------------------------------------------

class TestGroverCircuit:
    """Tests for grover_circuit()."""

    def test_returns_3qubit_circuit(self):
        circ = grover_circuit()
        assert circ.nqubits == 3

    def test_density_matrix_mode(self):
        circ = grover_circuit()
        dm = circ().state()
        assert dm.shape == (8, 8)

    def test_trace_one(self):
        circ = grover_circuit()
        dm = circ().state()
        assert abs(np.trace(dm) - 1.0) < 1e-6


class TestQftCircuit:
    """Tests for qft_circuit()."""

    def test_returns_3qubit_circuit(self):
        circ = qft_circuit()
        assert circ.nqubits == 3

    def test_density_matrix_mode(self):
        circ = qft_circuit()
        dm = circ().state()
        assert dm.shape == (8, 8)

    def test_trace_one(self):
        circ = qft_circuit()
        dm = circ().state()
        assert abs(np.trace(dm) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# evaluate_circuit
# ---------------------------------------------------------------------------

@pytest.fixture
def circuit_eval_setup(rb_setup):
    """Return (circuit, encoder, agent, noise_model) for evaluate_circuit tests."""
    circuit_gen, encoder, noise_model = rb_setup
    # Use a small random 1-qubit circuit so the fixture works with rb_setup
    circ = circuit_gen.generate_random_circuit()
    agent = _make_agent(encoder, noise_model)
    return circ, encoder, agent, noise_model


class TestEvaluateCircuit:
    """Tests for evaluate_circuit()."""

    def test_result_keys_full(self, circuit_eval_setup):
        circ, encoder, agent, noise_model = circuit_eval_setup
        res = evaluate_circuit(circ, encoder, agent, noise_model,
                               lambda_rb=0.9,
                               evaluate_mms=True, evaluate_no_noise=True)
        for key in ("n_qubits", "n_gates", "dm_truth", "dm_rl",
                    "dm_rb", "dm_no_noise", "dm_mms", "metrics"):
            assert key in res, f"Missing key: {key}"

    def test_rl_always_present(self, circuit_eval_setup):
        circ, encoder, agent, noise_model = circuit_eval_setup
        res = evaluate_circuit(circ, encoder, agent, noise_model,
                               lambda_rb=None, evaluate_mms=False, evaluate_no_noise=False)
        assert "rl" in res["metrics"]
        assert "dm_rl" in res

    def test_lambda_rb_none_skips_rb(self, circuit_eval_setup):
        circ, encoder, agent, noise_model = circuit_eval_setup
        res = evaluate_circuit(circ, encoder, agent, noise_model, lambda_rb=None)
        assert "rb" not in res["metrics"]
        assert "dm_rb" not in res

    def test_evaluate_mms_false(self, circuit_eval_setup):
        circ, encoder, agent, noise_model = circuit_eval_setup
        res = evaluate_circuit(circ, encoder, agent, noise_model,
                               lambda_rb=None, evaluate_mms=False)
        assert "mms" not in res["metrics"]

    def test_evaluate_no_noise_false(self, circuit_eval_setup):
        circ, encoder, agent, noise_model = circuit_eval_setup
        res = evaluate_circuit(circ, encoder, agent, noise_model,
                               lambda_rb=None, evaluate_no_noise=False)
        assert "no_noise" not in res["metrics"]

    def test_fidelity_in_range(self, circuit_eval_setup):
        circ, encoder, agent, noise_model = circuit_eval_setup
        res = evaluate_circuit(circ, encoder, agent, noise_model, lambda_rb=None)
        for model_key, m in res["metrics"].items():
            assert -0.01 <= m["fidelity"] <= 1.01, f"{model_key} fidelity out of range"
            assert m["trace"] >= -0.01, f"{model_key} trace distance negative"
            assert m["mse"] >= -1e-9, f"{model_key} MSE negative"

    def test_dm_shapes(self, circuit_eval_setup):
        circ, encoder, agent, noise_model = circuit_eval_setup
        res = evaluate_circuit(circ, encoder, agent, noise_model,
                               lambda_rb=0.9, evaluate_no_noise=True, evaluate_mms=True)
        dim = 2 ** res["n_qubits"]
        for key in ("dm_truth", "dm_rl", "dm_rb", "dm_no_noise", "dm_mms"):
            assert res[key].shape == (dim, dim), f"{key} has wrong shape"


class TestSummarizeCircuitMetrics:
    """Tests for summarize_circuit_metrics()."""

    def _make_results(self):
        return {
            "n_qubits": 1, "n_gates": 4,
            "dm_truth": np.eye(2, dtype=complex) / 2,
            "dm_rl":    np.eye(2, dtype=complex) / 2,
            "metrics": {
                "rl":       {"fidelity": 0.95, "trace": 0.05, "mse": 0.001},
                "no_noise": {"fidelity": 0.80, "trace": 0.20, "mse": 0.010},
            },
        }

    def test_returns_string(self):
        res = summarize_circuit_metrics(self._make_results())
        assert isinstance(res, str)

    def test_contains_model_labels(self):
        table = summarize_circuit_metrics(self._make_results())
        assert "RL" in table
        assert "No noise" in table

    def test_contains_metric_headers(self):
        table = summarize_circuit_metrics(self._make_results())
        assert "Fidelity" in table
        assert "Trace" in table
        assert "MSE" in table


class TestEvaluateOnDataset:
    """Tests for evaluate_on_dataset()."""

    def test_returns_expected_keys(self, circuit_eval_setup):
        circ, encoder, agent, noise_model = circuit_eval_setup
        # Build a minimal in-memory dataset (2 circuits)
        from rlnoise.dataset import DatasetGenerator
        from rlnoise.config import DatasetConfig
        cfg = DatasetConfig(n_circuits=2, moments=3, qubits=1,
                            primitive_gates=["rz", "rx"], clifford=False, mixed=False)
        ds = DatasetGenerator(cfg, noise_model.config).generate(verbose=False)
        results = evaluate_on_dataset(agent, ds.circuits, ds.labels, verbose=False)
        for key in ("per_circuit", "mean_fidelity", "std_fidelity",
                    "mean_trace_distance", "std_trace_distance",
                    "mean_mse", "std_mse", "n_circuits", "predicted_dms"):
            assert key in results, f"Missing key: {key}"

    def test_n_circuits_matches(self, circuit_eval_setup):
        circ, encoder, agent, noise_model = circuit_eval_setup
        from rlnoise.dataset import DatasetGenerator
        from rlnoise.config import DatasetConfig
        cfg = DatasetConfig(n_circuits=3, moments=3, qubits=1,
                            primitive_gates=["rz", "rx"], clifford=False, mixed=False)
        ds = DatasetGenerator(cfg, noise_model.config).generate(verbose=False)
        results = evaluate_on_dataset(agent, ds.circuits, ds.labels, verbose=False)
        assert results["n_circuits"] == 3
        assert len(results["per_circuit"]) == 3
        assert len(results["predicted_dms"]) == 3

    def test_per_circuit_keys(self, circuit_eval_setup):
        circ, encoder, agent, noise_model = circuit_eval_setup
        from rlnoise.dataset import DatasetGenerator
        from rlnoise.config import DatasetConfig
        cfg = DatasetConfig(n_circuits=2, moments=3, qubits=1,
                            primitive_gates=["rz", "rx"], clifford=False, mixed=False)
        ds = DatasetGenerator(cfg, noise_model.config).generate(verbose=False)
        results = evaluate_on_dataset(agent, ds.circuits, ds.labels, verbose=False)
        for r in results["per_circuit"]:
            assert "fidelity" in r
            assert "trace_distance" in r
            assert "mse" in r

    def test_fidelity_in_range(self, circuit_eval_setup):
        circ, encoder, agent, noise_model = circuit_eval_setup
        from rlnoise.dataset import DatasetGenerator
        from rlnoise.config import DatasetConfig
        cfg = DatasetConfig(n_circuits=4, moments=3, qubits=1,
                            primitive_gates=["rz", "rx"], clifford=False, mixed=False)
        ds = DatasetGenerator(cfg, noise_model.config).generate(verbose=False)
        results = evaluate_on_dataset(agent, ds.circuits, ds.labels, verbose=False)
        assert 0.0 <= results["mean_fidelity"] <= 1.0
        assert results["std_fidelity"] >= 0.0
        assert results["mean_trace_distance"] >= 0.0
        assert results["mean_mse"] >= 0.0

    def test_verbose_output(self, circuit_eval_setup, capsys):
        """verbose=True should print a summary block to stdout."""
        circ, encoder, agent, noise_model = circuit_eval_setup
        from rlnoise.dataset import DatasetGenerator
        from rlnoise.config import DatasetConfig
        cfg = DatasetConfig(n_circuits=12, moments=3, qubits=1,
                            primitive_gates=["rz", "rx"], clifford=False, mixed=False)
        ds = DatasetGenerator(cfg, noise_model.config).generate(verbose=False)
        results = evaluate_on_dataset(agent, ds.circuits, ds.labels, verbose=True)
        captured = capsys.readouterr()
        assert "Fidelity" in captured.out
        assert results["n_circuits"] == 12

