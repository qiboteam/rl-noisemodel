"""Visualization utilities for training results."""

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np


def plot_training_dashboard(
    results: Dict[str, Any],
    figsize: tuple = (12, 5),
    title: Optional[str] = None,
    filepath: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Plot training dashboard with reward and metric evolution.

    Two side-by-side plots are produced:
    - Left: reward vs. timestep for the training set and evaluation set.
    - Right: raw metric (e.g. trace distance or fidelity) vs. timestep for both sets.

    Shaded bands show ±1 standard deviation.

    Args:
        results: Dictionary returned by ``TrainingCallback.get_results()``.
            Must contain the keys ``timesteps``, ``train_results``,
            ``eval_results``, and ``metric_name``.  Each row of
            ``train_results``/``eval_results`` is
            ``[mean_reward, std_reward, mean_metric, std_metric]``.
        figsize: ``(width, height)`` in inches.
        title: Optional super-title for the figure.
        filepath: Optional path to save the figure (e.g. ``"plots/dashboard.png"``).
            If ``None``, the figure is not saved.

    Returns:
        The :class:`matplotlib.figure.Figure` object.
    """
    timesteps = np.array(results["timesteps"])
    train = np.array(results["train_results"])
    val = np.array(results["eval_results"])
    metric_name = results.get("metric_name", "metric")
    metric_label = metric_name.replace("_", " ").title()

    train_color = "#1f77b4"  # blue
    eval_color = "#ff7f0e"   # orange

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Reward ---
    ax = axes[0]
    ax.plot(timesteps, train[:, 0], color=train_color, label="Train")
    ax.fill_between(
        timesteps,
        train[:, 0] - train[:, 1],
        train[:, 0] + train[:, 1],
        alpha=0.2,
        color=train_color,
    )
    ax.plot(timesteps, val[:, 0], color=eval_color, label="Eval")
    ax.fill_between(
        timesteps,
        val[:, 0] - val[:, 1],
        val[:, 0] + val[:, 1],
        alpha=0.2,
        color=eval_color,
    )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Reward")
    ax.set_title("Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Raw metric ---
    ax = axes[1]
    ax.plot(timesteps, train[:, 2], color=train_color, label="Train")
    ax.fill_between(
        timesteps,
        train[:, 2] - train[:, 3],
        train[:, 2] + train[:, 3],
        alpha=0.2,
        color=train_color,
    )
    ax.plot(timesteps, val[:, 2], color=eval_color, label="Eval")
    ax.fill_between(
        timesteps,
        val[:, 2] - val[:, 3],
        val[:, 2] + val[:, 3],
        alpha=0.2,
        color=eval_color,
    )
    ax.set_xlabel("Timestep")
    ax.set_ylabel(metric_label)
    ax.set_title(metric_label)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if title is not None:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout()

    if filepath is not None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath)

    return fig


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

# Colours consistent with the published RB comparison style
_BENCH_COLORS = {
    "rl":       "#e60049",   # red
    "rb":       "#0bb4ff",   # blue
    "no_noise": "#2ca02c",   # green
    "mms":      "#ff7f0e",   # orange
}
_BENCH_LABELS = {
    "rl":       "RL model",
    "rb":       "Randomized benchmarking",
    "no_noise": "No noise",
    "mms":      "Maximally mixed state",
}


def plot_benchmarking_results(  # pylint: disable=too-many-locals
    results: Dict[str, Any],
    figsize: tuple = (18, 5),
    title: Optional[str] = None,
    filepath: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Plot the benchmarking comparison across four noise models.

    Three side-by-side subplots are produced showing fidelity (higher is
    better), trace distance, and MSE (both lower is better) as a function of
    circuit depth.  Each subplot contains four lines:

    * **RL model** – the trained RL agent.
    * **Randomized benchmarking** – uniform depolarizing channel fitted from RB.
    * **No noise** – noiseless simulation.
    * **Maximally mixed state** – maximally mixed baseline.

    Shaded bands show ±1 standard deviation.

    Args:
        results: Dictionary returned by
            :func:`~rlnoise.benchmarking.evaluate_benchmarks`.
            Must contain ``depths`` and model keys ``rl``, ``rb``,
            ``no_noise``, ``mms``, each with ``fidelity``, ``fidelity_std``,
            ``trace``, ``trace_std``, ``mse``, ``mse_std``.
        figsize: ``(width, height)`` in inches.
        title: Optional super-title for the figure.
        filepath: Optional path to save the figure. Parent directories are
            created automatically if they do not exist.

    Returns:
        The :class:`matplotlib.figure.Figure` object.
    """
    depths = np.array(results["depths"])
    model_keys = ["rl", "rb", "no_noise", "mms"]

    metrics = [
        ("fidelity", "Fidelity", True),       # (key, label, higher_is_better)
        ("trace",    "Trace Distance", False),
        ("mse",      "MSE", False),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)

    for ax, (metric_key, metric_label, _higher_better) in zip(axes, metrics):
        for k in model_keys:
            color = _BENCH_COLORS[k]
            label = _BENCH_LABELS[k]
            mean = np.array(results[k][metric_key])
            std  = np.array(results[k][f"{metric_key}_std"])

            ax.plot(depths, mean, color=color, label=label, linewidth=2, marker="o")
            ax.fill_between(depths, mean - std, mean + std, alpha=0.2, color=color)

        ax.set_xlabel("Circuit Depth")
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.set_xticks(depths)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    if title is not None:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout()

    if filepath is not None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath)

    return fig
