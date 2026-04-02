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
    train_results = [np.array([float(i), 0.1, 0.5 - i * 0.05, 0.02]) for i in range(1, n + 1)]
    eval_results = [np.array([float(i) * 0.9, 0.15, 0.4 - i * 0.04, 0.03]) for i in range(1, n + 1)]
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
        assert len(fig.axes) == 2  # reward + metric
        plt.close(fig)

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
