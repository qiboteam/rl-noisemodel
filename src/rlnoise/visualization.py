"""Visualization utilities for training and benchmarking results."""

from pathlib import Path
from typing import Any, Dict, List, Optional

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
    timesteps = np.array(results["timesteps"]) / 1_000
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
        ax.set_xlabel("Timesteps (×10³)")
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


def plot_rb_decay(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    rb_data: List[Dict[str, Any]],
    a: float,
    lambda_rb: float,
    figsize: tuple = (7, 4),
    title: Optional[str] = None,
    filepath: Optional[str] = None,
    color: str = "#0bb4ff",
) -> matplotlib.figure.Figure:
    """Plot the RB decay curve with empirical survival probabilities.

    Shows the empirical mean survival probability $P(|0\\rangle)$ at each
    depth together with the fitted exponential ``a * lambda^depth``.

    Args:
        rb_data: Output of :func:`~rlnoise.benchmarking.generate_rb_circuits`.
            Each entry must contain ``depth`` and ``qibo_circuits`` keys
            (survival probabilities are read directly from ``rb_data`` if a
            ``survival_prob`` key is present, otherwise the entry is skipped).
        a: Amplitude from the RB decay fit (returned by
            :func:`~rlnoise.benchmarking.fit_rb_decay`).
        lambda_rb: Decay constant from the RB decay fit.
        figsize: ``(width, height)`` in inches.
        title: Optional title for the figure.
        filepath: Optional path to save the figure.
        color: Line colour for the fitted curve.

    Returns:
        The :class:`matplotlib.figure.Figure` object.
    """
    depths = np.array([entry["depth"] for entry in rb_data])
    fit_curve = a * np.power(lambda_rb, depths)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        depths, fit_curve,
        "-o", color=color, linewidth=2,
        label=f"Fit: {a:.3f} \u00b7 {lambda_rb:.4f}\u1d48",
    )
    ax.set_xlabel("Circuit depth")
    ax.set_ylabel("Survival probability $P(|0\\rangle)$")
    ax.set_title(title or "RB Decay Fit")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if filepath is not None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath)

    return fig


def plot_benchmarking_results(  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments,too-many-branches
    results: Dict[str, Any],
    figsize: tuple = None,
    title: Optional[str] = None,
    filepath: Optional[str] = None,
    show_fidelity: bool = True,
    show_trace: bool = True,
    show_mse: bool = True,
) -> matplotlib.figure.Figure:
    """Plot the benchmarking comparison across active noise models.

    Up to three side-by-side subplots are produced (fidelity, trace distance,
    MSE) controlled by the ``show_*`` flags.  A curve is drawn only for
    models whose key is present in *results*, so models that were skipped
    in :func:`~rlnoise.benchmarking.evaluate_benchmarks` are automatically
    omitted.

    Shaded bands show \u00b11 standard deviation.

    Args:
        results: Dictionary returned by
            :func:`~rlnoise.benchmarking.evaluate_benchmarks`.
            Must contain ``depths`` and at least one of the model keys
            ``rl``, ``rb``, ``no_noise``, ``mms``.
        figsize: ``(width, height)`` in inches.  Defaults to 6 inches per
            enabled panel.
        title: Optional super-title for the figure.
        filepath: Optional path to save the figure. Parent directories are
            created automatically if they do not exist.
        show_fidelity: Whether to include the fidelity panel.
        show_trace: Whether to include the trace-distance panel.
        show_mse: Whether to include the MSE panel.

    Returns:
        The :class:`matplotlib.figure.Figure` object.

    Raises:
        ValueError: If all metric panels are disabled.
    """
    depths = np.array(results["depths"])
    # Only draw curves for models that are present in the results dict
    model_keys = [k for k in ("rl", "rb", "no_noise", "mms") if k in results]

    active_metrics = []
    if show_fidelity:
        active_metrics.append(("fidelity", "Fidelity"))
    if show_trace:
        active_metrics.append(("trace", "Trace Distance"))
    if show_mse:
        active_metrics.append(("mse", "MSE"))

    if not active_metrics:
        raise ValueError(
            "At least one of show_fidelity, show_trace, show_mse must be True."
        )

    n_panels = len(active_metrics)
    if figsize is None:
        figsize = (6 * n_panels, 5)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    for ax, (metric_key, metric_label) in zip(axes, active_metrics):
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


# ---------------------------------------------------------------------------
# Single-circuit visualisations
# ---------------------------------------------------------------------------

#: Model key → display colour (shared with benchmarking plots)
_CIRC_COLORS = {
    "truth":    "#e60049",  # red  — ground truth
    "rl":       "#0bb4ff",  # blue
    "rb":       "#50e991",  # green
    "no_noise": "#9b19f5",  # purple
    "mms":      "#ffa300",  # amber
}
_CIRC_LABELS = {
    "truth":    "Ground truth",
    "rl":       "RL",
    "rb":       "RB",
    "no_noise": "No noise",
    "mms":      "MMS",
}


def plot_shots(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches
    circuit_results: dict,
    show_truth: bool = True,
    show_rl: bool = True,
    show_rb: bool = True,
    show_no_noise: bool = True,
    show_mms: bool = False,
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
    filepath: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Bar chart of basis-state probabilities for each active model.

    The diagonal of each density matrix gives the classical probability of
    measuring each computational basis state.  A grouped bar chart is drawn
    with one group per basis state and one bar per active model.

    Args:
        circuit_results: Dictionary returned by
            :func:`~rlnoise.benchmarking.evaluate_circuit`.
        show_truth: Whether to include the ground-truth bars.
        show_rl: Whether to include the RL model bars.
        show_rb: Whether to include the RB model bars (skipped automatically
            when ``dm_rb`` is absent from *circuit_results*).
        show_no_noise: Whether to include the no-noise bars.
        show_mms: Whether to include the MMS bars.
        figsize: ``(width, height)`` in inches.  Defaults to auto.
        title: Optional figure title.
        filepath: Optional path to save the figure.

    Returns:
        The :class:`matplotlib.figure.Figure` object.
    """
    n_qubits = circuit_results["n_qubits"]
    dim = 2 ** n_qubits
    labels_x = [format(i, f"0{n_qubits}b") for i in range(dim)]

    # Collect active (key, DM) pairs in display order
    candidates = [
        ("truth",    "dm_truth",    show_truth),
        ("rl",       "dm_rl",       show_rl),
        ("rb",       "dm_rb",       show_rb),
        ("no_noise", "dm_no_noise", show_no_noise),
        ("mms",      "dm_mms",      show_mms),
    ]
    active = [
        (key, circuit_results[dm_key])
        for key, dm_key, flag in candidates
        if flag and dm_key in circuit_results
    ]
    if not active:
        raise ValueError("At least one model must be selected and present in circuit_results.")

    n_models = len(active)
    bar_w = 0.8 / n_models
    x = np.arange(dim)

    if figsize is None:
        figsize = (max(10, dim * 1.2), 5)

    fig, ax = plt.subplots(figsize=figsize)
    for i, (key, dm) in enumerate(active):
        probs = np.real(np.diag(dm))
        offset = (i - (n_models - 1) / 2) * bar_w
        ax.bar(
            x + offset, probs, width=bar_w,
            color=_CIRC_COLORS[key],
            label=_CIRC_LABELS[key],
        )

    ax.set_xlabel("Basis state")
    ax.set_ylabel("Probability")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, rotation=45 if dim > 8 else 0)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(title or "Basis-state probabilities")
    fig.tight_layout()

    if filepath is not None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath)

    return fig


def plot_density_matrix_heatmap(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches
    circuit_results: dict,
    show_rl: bool = True,
    show_rb: bool = True,
    show_no_noise: bool = False,
    show_mms: bool = False,
    cmap: str = "plasma",
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
    filepath: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Heatmap of the element-wise absolute difference |ρ_truth − ρ_model|.

    Each active model is shown as a separate panel.  All panels share the same
    colour scale so comparisons are straightforward.

    Args:
        circuit_results: Dictionary returned by
            :func:`~rlnoise.benchmarking.evaluate_circuit`.
        show_rl: Whether to include the RL model panel.
        show_rb: Whether to include the RB model panel.
        show_no_noise: Whether to include the no-noise panel.
        show_mms: Whether to include the MMS panel.
        cmap: Matplotlib colour map name (default ``"plasma"``).
        figsize: ``(width, height)`` in inches.  Defaults to
            ``(6 * n_panels, 5)``.
        title: Optional super-title for the figure.
        filepath: Optional path to save the figure.

    Returns:
        The :class:`matplotlib.figure.Figure` object.
    """
    dm_truth = circuit_results["dm_truth"]

    candidates = [
        ("rl",       "dm_rl",       show_rl),
        ("rb",       "dm_rb",       show_rb),
        ("no_noise", "dm_no_noise", show_no_noise),
        ("mms",      "dm_mms",      show_mms),
    ]
    active = [
        (key, circuit_results[dm_key])
        for key, dm_key, flag in candidates
        if flag and dm_key in circuit_results
    ]
    if not active:
        raise ValueError("At least one model must be selected and present in circuit_results.")

    n_panels = len(active)
    if figsize is None:
        figsize = (6 * n_panels, 5)

    # Compute diffs to find shared colour scale
    diffs = [np.abs(dm_truth - dm) for _, dm in active]
    vmin = min(d.min() for d in diffs)
    vmax = max(d.max() for d in diffs)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    im = None
    for ax, (key, _), diff in zip(axes, active, diffs):
        im = ax.imshow(diff, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"|ρ_truth − ρ_{_CIRC_LABELS[key]}|")
        ax.set_xticks([])
        ax.set_yticks([])

    if im is not None:
        fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.046, pad=0.04)

    if title is not None:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout()

    if filepath is not None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath)

    return fig


def plot_circuit_metrics(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches
    circuit_results: dict,
    show_fidelity: bool = True,
    show_trace: bool = True,
    show_mse: bool = True,
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
    filepath: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Grouped bar chart of fidelity, trace distance and MSE per noise model.

    Up to three side-by-side panels are shown (controlled by the ``show_*``
    flags).  Each panel contains one bar per active model, coloured
    consistently with the other benchmarking plots.

    Args:
        circuit_results: Dictionary returned by
            :func:`~rlnoise.benchmarking.evaluate_circuit`.
        show_fidelity: Whether to include the fidelity panel.
        show_trace: Whether to include the trace-distance panel.
        show_mse: Whether to include the MSE panel.
        figsize: ``(width, height)`` in inches.  Defaults to
            ``(5 * n_panels, 4)``.
        title: Optional super-title for the figure.
        filepath: Optional path to save the figure.

    Returns:
        The :class:`matplotlib.figure.Figure` object.

    Raises:
        ValueError: If all metric panels are disabled.
    """
    metrics = circuit_results["metrics"]
    model_keys = [k for k in ("rl", "rb", "no_noise", "mms") if k in metrics]

    active_panels: List[tuple] = []
    if show_fidelity:
        active_panels.append(("fidelity", "Fidelity"))
    if show_trace:
        active_panels.append(("trace", "Trace Distance"))
    if show_mse:
        active_panels.append(("mse", "MSE"))

    if not active_panels:
        raise ValueError("At least one of show_fidelity, show_trace, show_mse must be True.")

    n_panels = len(active_panels)
    if figsize is None:
        figsize = (5 * n_panels, 4)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    x = np.arange(len(model_keys))
    bar_w = 0.6
    for ax, (metric_key, metric_label) in zip(axes, active_panels):
        values = [metrics[k][metric_key] for k in model_keys]
        colors = [_CIRC_COLORS[k] for k in model_keys]
        xlabels = [_CIRC_LABELS[k] for k in model_keys]

        bars = ax.bar(x, values, width=bar_w, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=15, ha="right")
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)

        # Annotate bars with numeric values
        for rect, val in zip(bars, values):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() * 1.01,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=8,
            )

    if title is not None:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout()

    if filepath is not None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath)

    return fig
