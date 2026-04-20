"""Tests for visualization utilities."""

import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.figure

from rlnoise.visualization import plot_training_dashboard, plot_benchmarking_results


@pytest.fixture
def training_results():
    """Synthetic training results compatible with TrainingCallback.get_results()."""
    n = 5
    timesteps = list(range(100, 100 * (n + 1), 100))
    train_results = [np.array([float(i), 0.1, 0.5 - i * 0.05, 0.02, 0.5 + i * 0.05, 0.02]) for i in range(1, n + 1)]
    eval_results = [np.array([float(i) * 0.9, 0.15, 0.4 - i * 0.04, 0.03, 0.6 + i * 0.04, 0.02]) for i in range(1, n + 1)]
    return {
        "timesteps": timesteps,
        "train_results": train_results,
        "eval_results": eval_results,
        "best_mean_reward": 5.0,
        "metric_name": "trace",
    }


@pytest.fixture
def benchmarking_results():
    """Synthetic benchmarking results compatible with evaluate_benchmarks()."""
    depths = [3, 5, 7]
    n = len(depths)

    def make_model():
        return {
            "fidelity": [0.9 - 0.05 * i for i in range(n)],
            "fidelity_std": [0.02] * n,
            "trace": [0.05 + 0.02 * i for i in range(n)],
            "trace_std": [0.01] * n,
            "mse": [0.01 + 0.005 * i for i in range(n)],
            "mse_std": [0.002] * n,
        }

    return {
        "depths": depths,
        "rl": make_model(),
        "rb": make_model(),
        "no_noise": make_model(),
        "mms": make_model(),
    }


class TestPlotTrainingDashboard:
    """Tests for plot_training_dashboard."""

    def test_returns_figure(self, training_results):
        fig = plot_training_dashboard(training_results)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_correct_number_of_axes(self, training_results):
        fig = plot_training_dashboard(training_results)
        assert len(fig.axes) == 3  # reward + trace distance + fidelity
        plt.close(fig)

    def test_selective_panels(self, training_results):
        fig = plot_training_dashboard(training_results, show_reward=True, show_trace_distance=False, show_fidelity=False)
        assert len(fig.axes) == 1
        plt.close(fig)

        fig = plot_training_dashboard(training_results, show_reward=True, show_trace_distance=True, show_fidelity=False)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_no_panels_raises(self, training_results):
        with pytest.raises(ValueError):
            plot_training_dashboard(training_results, show_reward=False, show_trace_distance=False, show_fidelity=False)

    def test_with_title(self, training_results):
        fig = plot_training_dashboard(training_results, title="Test Title")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_with_filepath(self, training_results, tmp_path):
        filepath = str(tmp_path / "dashboard.png")
        fig = plot_training_dashboard(training_results, filepath=filepath)
        import os
        assert os.path.exists(filepath)
        plt.close(fig)

    def test_custom_figsize(self, training_results):
        fig = plot_training_dashboard(training_results, figsize=(8, 4))
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_short_columns_hides_panel(self, training_results):
        """Guard branch: panel with column index ≥ array width is hidden."""
        # Create results with only 2 columns (reward mean/std only)
        short = dict(training_results)
        short["train_results"] = [r[:2] for r in training_results["train_results"]]
        short["eval_results"]  = [r[:2] for r in training_results["eval_results"]]
        # Requesting fidelity (col index 4) on a 2-column array triggers the guard
        fig = plot_training_dashboard(short, show_reward=True,
                                      show_trace_distance=True, show_fidelity=True)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


class TestPlotBenchmarkingResults:
    """Tests for plot_benchmarking_results."""

    def test_returns_figure(self, benchmarking_results):
        fig = plot_benchmarking_results(benchmarking_results)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_correct_number_of_axes(self, benchmarking_results):
        fig = plot_benchmarking_results(benchmarking_results)
        assert len(fig.axes) == 3  # fidelity + trace + mse
        plt.close(fig)

    def test_with_title(self, benchmarking_results):
        fig = plot_benchmarking_results(benchmarking_results, title="Benchmark")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_with_filepath(self, benchmarking_results, tmp_path):
        filepath = str(tmp_path / "bench.png")
        fig = plot_benchmarking_results(benchmarking_results, filepath=filepath)
        import os
        assert os.path.exists(filepath)
        plt.close(fig)

    def test_custom_figsize(self, benchmarking_results):
        fig = plot_benchmarking_results(benchmarking_results, figsize=(12, 4))
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Single-circuit plot tests
# ---------------------------------------------------------------------------

@pytest.fixture
def circuit_results_1q():
    """Minimal evaluate_circuit output for a 1-qubit circuit."""
    dm = np.array([[0.7, 0.1 + 0.0j], [0.1 + 0.0j, 0.3]], dtype=complex)
    dm_rl = np.array([[0.68, 0.09j], [-0.09j, 0.32]], dtype=complex)
    dm_rb = np.array([[0.65, 0.08j], [-0.08j, 0.35]], dtype=complex)
    dm_no = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    dm_mms = np.eye(2, dtype=complex) / 2
    return {
        "n_qubits": 1,
        "n_gates": 4,
        "dm_truth":    dm,
        "dm_rl":       dm_rl,
        "dm_rb":       dm_rb,
        "dm_no_noise": dm_no,
        "dm_mms":      dm_mms,
        "metrics": {
            "rl":       {"fidelity": 0.95, "trace": 0.05, "mse": 0.001},
            "rb":       {"fidelity": 0.88, "trace": 0.10, "mse": 0.005},
            "no_noise": {"fidelity": 0.80, "trace": 0.20, "mse": 0.010},
            "mms":      {"fidelity": 0.50, "trace": 0.50, "mse": 0.100},
        },
    }


class TestPlotShots:
    """Tests for plot_shots()."""

    def test_returns_figure(self, circuit_results_1q):
        from rlnoise.visualization import plot_shots
        fig = plot_shots(circuit_results_1q)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_no_rb_when_absent(self, circuit_results_1q):
        from rlnoise.visualization import plot_shots
        del circuit_results_1q["dm_rb"]
        fig = plot_shots(circuit_results_1q, show_rb=True)  # flag set but key absent
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_show_mms(self, circuit_results_1q):
        from rlnoise.visualization import plot_shots
        fig = plot_shots(circuit_results_1q, show_mms=True)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_raises_when_nothing_to_show(self, circuit_results_1q):
        from rlnoise.visualization import plot_shots
        with pytest.raises(ValueError):
            plot_shots(circuit_results_1q,
                       show_truth=False, show_rl=False, show_rb=False,
                       show_no_noise=False, show_mms=False)

    def test_saves_file(self, circuit_results_1q, tmp_path):
        from rlnoise.visualization import plot_shots
        filepath = str(tmp_path / "shots.png")
        fig = plot_shots(circuit_results_1q, filepath=filepath)
        plt.close(fig)
        assert (tmp_path / "shots.png").exists()

    def test_custom_title(self, circuit_results_1q):
        from rlnoise.visualization import plot_shots
        fig = plot_shots(circuit_results_1q, title="My Circuit Shots")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


class TestPlotDensityMatrixHeatmap:
    """Tests for plot_density_matrix_heatmap()."""

    def test_returns_figure(self, circuit_results_1q):
        from rlnoise.visualization import plot_density_matrix_heatmap
        fig = plot_density_matrix_heatmap(circuit_results_1q)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_single_panel(self, circuit_results_1q):
        from rlnoise.visualization import plot_density_matrix_heatmap
        fig = plot_density_matrix_heatmap(circuit_results_1q,
                                          show_rl=True, show_rb=False)
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_no_panels_raises(self, circuit_results_1q):
        from rlnoise.visualization import plot_density_matrix_heatmap
        with pytest.raises(ValueError):
            plot_density_matrix_heatmap(circuit_results_1q,
                                        show_rl=False, show_rb=False,
                                        show_no_noise=False, show_mms=False)

    def test_custom_cmap(self, circuit_results_1q):
        from rlnoise.visualization import plot_density_matrix_heatmap
        fig = plot_density_matrix_heatmap(circuit_results_1q, cmap="viridis")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_saves_file(self, circuit_results_1q, tmp_path):
        from rlnoise.visualization import plot_density_matrix_heatmap
        filepath = str(tmp_path / "heatmap.png")
        fig = plot_density_matrix_heatmap(circuit_results_1q, filepath=filepath)
        plt.close(fig)
        assert (tmp_path / "heatmap.png").exists()

    def test_absent_model_skipped_silently(self, circuit_results_1q):
        from rlnoise.visualization import plot_density_matrix_heatmap
        del circuit_results_1q["dm_rb"]
        fig = plot_density_matrix_heatmap(circuit_results_1q,
                                          show_rl=True, show_rb=True)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_with_title(self, circuit_results_1q):
        from rlnoise.visualization import plot_density_matrix_heatmap
        fig = plot_density_matrix_heatmap(circuit_results_1q, title="My Heatmap")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


class TestPlotCircuitMetrics:
    """Tests for plot_circuit_metrics()."""

    def test_returns_figure(self, circuit_results_1q):
        from rlnoise.visualization import plot_circuit_metrics
        fig = plot_circuit_metrics(circuit_results_1q)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_three_panels(self, circuit_results_1q):
        from rlnoise.visualization import plot_circuit_metrics
        fig = plot_circuit_metrics(circuit_results_1q,
                                   show_fidelity=True, show_trace=True, show_mse=True)
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_single_panel(self, circuit_results_1q):
        from rlnoise.visualization import plot_circuit_metrics
        fig = plot_circuit_metrics(circuit_results_1q,
                                   show_fidelity=True, show_trace=False, show_mse=False)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_no_panels_raises(self, circuit_results_1q):
        from rlnoise.visualization import plot_circuit_metrics
        with pytest.raises(ValueError):
            plot_circuit_metrics(circuit_results_1q,
                                 show_fidelity=False, show_trace=False, show_mse=False)

    def test_saves_file(self, circuit_results_1q, tmp_path):
        from rlnoise.visualization import plot_circuit_metrics
        filepath = str(tmp_path / "metrics.png")
        fig = plot_circuit_metrics(circuit_results_1q, filepath=filepath)
        plt.close(fig)
        assert (tmp_path / "metrics.png").exists()

    def test_with_title(self, circuit_results_1q):
        from rlnoise.visualization import plot_circuit_metrics
        fig = plot_circuit_metrics(circuit_results_1q, title="My Metrics")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

