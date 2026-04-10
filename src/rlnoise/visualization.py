"""Visualization utilities for training results."""

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np


def plot_training_dashboard(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    results: Dict[str, Any],
    figsize: tuple = None,
    title: Optional[str] = None,
    filepath: Optional[str] = None,
    show_reward: bool = True,
    show_trace_distance: bool = True,
    show_fidelity: bool = True,
) -> matplotlib.figure.Figure:
    """Plot training dashboard with selectable metrics.

    Up to three side-by-side plots are produced based on the boolean flags:
    - Reward vs. timestep (train and eval).
    - Trace distance vs. timestep (train and eval).
    - Fidelity vs. timestep (train and eval).

    Shaded bands show ±1 standard deviation.

    Args:
        results: Dictionary returned by ``TrainingCallback.get_results()``.
            Must contain ``timesteps``, ``train_results``, ``eval_results``.
            Each row of ``train_results``/``eval_results`` is
            ``[mean_reward, std_reward, mean_trace_dist, std_trace_dist,
            mean_fidelity, std_fidelity]``.
        figsize: ``(width, height)`` in inches.  Defaults to 6 inches per
            enabled panel.
        title: Optional super-title for the figure.
        filepath: Optional path to save the figure.
        show_reward: Whether to include the reward panel.
        show_trace_distance: Whether to include the trace-distance panel.
        show_fidelity: Whether to include the fidelity panel.

    Returns:
        The :class:`matplotlib.figure.Figure` object.
    """
    timesteps = np.array(results["timesteps"])
    train = np.array(results["train_results"])
    val = np.array(results["eval_results"])

    train_color = "#1f77b4"  # blue
    eval_color = "#ff7f0e"   # orange

    panels = []
    if show_reward:
        panels.append(("Reward", 0, 1))
    if show_trace_distance:
        panels.append(("Trace Distance", 2, 3))
    if show_fidelity:
        panels.append(("Fidelity", 4, 5))

    n_panels = len(panels)
    if n_panels == 0:
        raise ValueError(
            "At least one of show_reward, show_trace_distance, show_fidelity must be True."
        )

    if figsize is None:
        figsize = (6 * n_panels, 5)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    for ax, (label, mean_col, std_col) in zip(axes, panels):
        # Guard against results arrays that lack fidelity columns (old files)
        if mean_col >= train.shape[1]:
            ax.set_visible(False)
            continue
        ax.plot(timesteps, train[:, mean_col], color=train_color, label="Train")
        ax.fill_between(
            timesteps,
            train[:, mean_col] - train[:, std_col],
            train[:, mean_col] + train[:, std_col],
            alpha=0.2,
            color=train_color,
        )
        ax.plot(timesteps, val[:, mean_col], color=eval_color, label="Eval")
        ax.fill_between(
            timesteps,
            val[:, mean_col] - val[:, std_col],
            val[:, mean_col] + val[:, std_col],
            alpha=0.2,
            color=eval_color,
        )
        ax.set_xlabel("Timestep")
        ax.set_ylabel(label)
        ax.set_title(label)
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
