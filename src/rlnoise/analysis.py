"""Explainable-AI analysis of the trained RL agent's noise actions.

This module lets you inspect *what* noise the agent actually applies to a
dataset of circuits, providing a window into the noise the agent has learned.

Main workflow::

    from rlnoise.analysis import collect_actions, noise_summary
    from rlnoise.analysis import (
        plot_noise_distributions,
        plot_noise_by_gate,
        plot_noise_by_qubit,
        plot_spatial_noise,
        plot_noise_correlation,
    )

    data = collect_actions(agent, circuits, verbose=True)
    print(noise_summary(data))
    plot_noise_distributions(data)

The :func:`collect_actions` function is the entry point.  It runs the agent
over every circuit in the supplied array and records the scaled noise value
the agent assigns to each (moment, qubit) position.  All subsequent plotting
helpers accept the dict it returns.

Noise channel ordering (matching ``QuantumCircuitEnv._apply_action``)
----------------------------------------------------------------------
- ``epsilon_x``  — coherent X rotation error
- ``epsilon_z``  — coherent Z rotation error
- ``reset``      — amplitude-damping / reset probability
- ``depol``      — depolarising parameter λ

Gate-type labels
----------------
``"rx"``, ``"rz"``, ``"cz"``, ``"id"`` (identity / no gate on that qubit).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from rlnoise.circuit_encoder import CircuitEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOISE_CHANNELS: List[str] = ["epsilon_x", "epsilon_z", "reset", "depol"]

NOISE_LABELS: Dict[str, str] = {
    "epsilon_x": "Coherent X  (ε_x)",
    "epsilon_z": "Coherent Z  (ε_z)",
    "reset":     "Damping / Reset",
    "depol":     "Depolarising (λ)",
}

NOISE_COLORS: Dict[str, str] = {
    "epsilon_x": "#0bb4ff",
    "epsilon_z": "#e60049",
    "reset":     "#50e991",
    "depol":     "#9b19f5",
}

GATE_COLORS: Dict[str, str] = {
    "rx":  "#0bb4ff",
    "rz":  "#e60049",
    "cz":  "#f46a9b",
    "id":  "#bbbbbb",
}

# Encoding-array column indices (from CircuitEncoder)
_IDX: Dict[str, int] = {
    "epsilon_x": CircuitEncoder.IDX_EPSILON_X,  # 7
    "epsilon_z": CircuitEncoder.IDX_EPSILON_Z,  # 6
    "reset":     CircuitEncoder.IDX_RESET,       # 5
    "depol":     CircuitEncoder.IDX_DEPOL,       # 4
    "rx":        CircuitEncoder.IDX_RX,          # 1
    "rz":        CircuitEncoder.IDX_RZ,          # 0
    "cz":        CircuitEncoder.IDX_CZ,          # 2
}


# ---------------------------------------------------------------------------
# Core data-collection
# ---------------------------------------------------------------------------

def collect_actions(  # pylint: disable=too-many-locals
    rl_agent,
    circuits: np.ndarray,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the agent over *circuits* and record every noise action it takes.

    For each circuit the agent is run deterministically.  The noise values
    written into the circuit array at every (moment, qubit) position are
    extracted and stored.

    Args:
        rl_agent: Trained :class:`~rlnoise.rl_agent.RLAgent`.
        circuits: Circuit arrays, shape
            ``(n_circuits, n_moments, n_qubits, encoding_dim)``.
        verbose: Print progress every ~10 % of circuits.

    Returns:
        Dictionary with the following keys:

        - ``"epsilon_x"``  — ndarray ``(n_circuits, n_moments, n_qubits)``
        - ``"epsilon_z"``  — ndarray  (same shape)
        - ``"reset"``      — ndarray  (same shape)
        - ``"depol"``      — ndarray  (same shape)
        - ``"gate_type"``  — object ndarray ``(n_circuits, n_moments, n_qubits)``
          with values ``"rx"``, ``"rz"``, ``"cz"``, or ``"id"``
        - ``"n_circuits"``, ``"n_moments"``, ``"n_qubits"`` — ints
    """
    n_circuits, n_moments, n_qubits, _ = circuits.shape

    noise_arrays = {ch: np.zeros((n_circuits, n_moments, n_qubits), dtype=np.float32)
                    for ch in NOISE_CHANNELS}
    gate_type = np.empty((n_circuits, n_moments, n_qubits), dtype=object)

    log_step = max(1, n_circuits // 10)

    for i in range(n_circuits):
        if verbose and i % log_step == 0:
            print(f"  Collecting actions: circuit {i + 1}/{n_circuits}…")

        circuit_arr = circuits[i]  # (n_moments, n_qubits, encoding_dim)

        # ── Determine gate types from original (pre-noise) array ──────────
        for m in range(n_moments):
            for q in range(n_qubits):
                enc = circuit_arr[m, q]
                if enc[_IDX["rx"]] > 0:
                    gate_type[i, m, q] = "rx"
                elif enc[_IDX["rz"]] > 0:
                    gate_type[i, m, q] = "rz"
                elif enc[_IDX["cz"]] > 0:
                    gate_type[i, m, q] = "cz"
                else:
                    gate_type[i, m, q] = "id"

        # ── Apply agent and read back the written noise values ────────────
        noisy_arr = rl_agent.apply_to_circuit(circuit_arr, return_qibo=False)
        # noisy_arr shape: (n_moments, n_qubits, encoding_dim)

        for ch, idx in (
            ("epsilon_x", _IDX["epsilon_x"]),
            ("epsilon_z", _IDX["epsilon_z"]),
            ("reset",     _IDX["reset"]),
            ("depol",     _IDX["depol"]),
        ):
            noise_arrays[ch][i] = noisy_arr[:, :, idx]

    if verbose:
        print("  Done.")

    return {
        **noise_arrays,
        "gate_type": gate_type,
        "n_circuits": n_circuits,
        "n_moments": n_moments,
        "n_qubits": n_qubits,
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def noise_summary(  # pylint: disable=too-many-locals
    actions_data: Dict[str, Any],
    skip_identity: bool = True,
    skip_zero: bool = True,
) -> str:
    """Return a formatted table of mean ± std for each noise channel.

    Args:
        actions_data: Output of :func:`collect_actions`.
        skip_identity: If ``True``, exclude (moment, qubit) positions where
            no gate was applied (gate_type == ``"id"``).
        skip_zero: If ``True``, exclude positions where the agent output
            exactly zero for the channel being summarised.

    Returns:
        Multi-line string table.
    """
    base_mask = np.ones(
        (actions_data["n_circuits"], actions_data["n_moments"], actions_data["n_qubits"]),
        dtype=bool,
    )
    if skip_identity:
        base_mask = actions_data["gate_type"] != "id"

    rows = []
    for ch in NOISE_CHANNELS:
        ch_mask = base_mask.copy()
        if skip_zero:
            ch_mask &= actions_data[ch] != 0.0
        vals = actions_data[ch][ch_mask]
        if vals.size == 0:
            rows.append((NOISE_LABELS[ch], float("nan"), float("nan"),
                         float("nan"), float("nan")))
        else:
            rows.append((NOISE_LABELS[ch], vals.mean(), vals.std(),
                         vals.min(), vals.max()))

    col_w = 22
    header = (f"{'Channel':<{col_w}} {'Mean':>10} {'Std':>10} "
              f"{'Min':>10} {'Max':>10}")
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for label, mean, std, mn, mx in rows:
        lines.append(
            f"{label:<{col_w}} {mean:>10.5f} {std:>10.5f} {mn:>10.5f} {mx:>10.5f}"
        )
    lines.append(sep)
    skip_parts = []
    if skip_identity:
        skip_parts.append("identity excluded")
    if skip_zero:
        skip_parts.append("zeros excluded")
    skip_note = f" ({', '.join(skip_parts)})" if skip_parts else ""
    lines.append(f"Circuits: {actions_data['n_circuits']}  "
                 f"Moments: {actions_data['n_moments']}  "
                 f"Qubits: {actions_data['n_qubits']}{skip_note}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting helpers (internal)
# ---------------------------------------------------------------------------

def _save_and_return(fig: plt.Figure, filepath: Optional[str]) -> plt.Figure:  # pragma: no cover
    if filepath is not None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
    return fig


def _flat_values(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    actions_data: Dict[str, Any],
    channel: str,
    gate_filter: Optional[str] = None,
    qubit_filter: Optional[int] = None,
    skip_identity: bool = True,
    skip_zero: bool = True,
) -> np.ndarray:
    """Return flat array of noise values with optional filtering."""
    vals = actions_data[channel]      # (n_c, n_m, n_q)
    gt   = actions_data["gate_type"]  # (n_c, n_m, n_q)

    mask = np.ones_like(vals, dtype=bool)
    if skip_identity:
        mask &= gt != "id"
    if gate_filter is not None:
        mask &= gt == gate_filter
    if qubit_filter is not None:
        qubit_mask = np.zeros_like(vals, dtype=bool)
        qubit_mask[:, :, qubit_filter] = True
        mask &= qubit_mask
    if skip_zero:
        mask &= vals != 0.0

    return vals[mask].ravel()


# ---------------------------------------------------------------------------
# Plot 1 — Global noise distributions
# ---------------------------------------------------------------------------

def plot_noise_distributions(  # pragma: no cover  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    actions_data: Dict[str, Any],
    skip_identity: bool = True,
    skip_zero: bool = True,
    bins: int = 40,
    figsize: Optional[tuple] = None,
    filepath: Optional[str] = None,
) -> plt.Figure:
    """Plot one histogram per noise channel across all circuits and qubits.

    Args:
        actions_data: Output of :func:`collect_actions`.
        skip_identity: Exclude identity-gate positions.
        skip_zero: Exclude positions where the agent output zero.
        bins: Number of histogram bins.
        figsize: Figure size; defaults to ``(14, 4)``.
        filepath: Save path for the figure (optional).

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    if figsize is None:
        figsize = (14, 4)

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    for ax, ch in zip(axes, NOISE_CHANNELS):
        vals = _flat_values(actions_data, ch, skip_identity=skip_identity,
                            skip_zero=skip_zero)
        mean, std = vals.mean(), vals.std()
        color = NOISE_COLORS[ch]

        ax.hist(vals, bins=bins, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(mean, color="black", linewidth=1.8, linestyle="--",
                   label=f"μ = {mean:.4f}")
        ax.axvspan(mean - std, mean + std, alpha=0.15, color="black")
        ax.set_xlabel(NOISE_LABELS[ch])
        ax.set_ylabel("Count")
        ax.set_title(NOISE_LABELS[ch])
        ax.legend(fontsize=9)

    skip_parts = []
    if skip_identity:
        skip_parts.append("identity excluded")
    if skip_zero:
        skip_parts.append("zeros excluded")
    skip_note = f" ({', '.join(skip_parts)})" if skip_parts else ""
    n_c = actions_data["n_circuits"]
    fig.suptitle(f"Agent noise distributions — {n_c} circuits{skip_note}", fontsize=13)
    fig.tight_layout()
    return _save_and_return(fig, filepath)


# ---------------------------------------------------------------------------
# Plot 2 — Noise breakdown by gate type
# ---------------------------------------------------------------------------

def plot_noise_by_gate(  # pragma: no cover  # pylint: disable=too-many-locals
    actions_data: Dict[str, Any],
    skip_zero: bool = True,
    bins: int = 30,
    figsize: Optional[tuple] = None,
    filepath: Optional[str] = None,
) -> plt.Figure:
    """Plot noise distributions split by gate type (RX / RZ / CZ).

    Each row corresponds to a noise channel; each column to a gate type
    present in the dataset.

    Args:
        actions_data: Output of :func:`collect_actions`.
        skip_zero: Exclude positions where the agent output zero.
        bins: Number of histogram bins.
        figsize: Figure size; auto-computed if ``None``.
        filepath: Save path for the figure (optional).

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    present_gates = [g for g in ("rx", "rz", "cz")
                     if np.any(actions_data["gate_type"] == g)]
    if not present_gates:
        raise ValueError("No named gates found in actions_data.")

    n_rows = len(NOISE_CHANNELS)
    n_cols = len(present_gates)
    if figsize is None:
        figsize = (4.5 * n_cols, 3.5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for r, ch in enumerate(NOISE_CHANNELS):
        for c, gate in enumerate(present_gates):
            ax = axes[r][c]
            vals = _flat_values(actions_data, ch, gate_filter=gate,
                                skip_identity=True, skip_zero=skip_zero)
            if vals.size == 0:
                ax.set_visible(False)
                continue
            mean, std = vals.mean(), vals.std()
            ax.hist(vals, bins=bins, color=GATE_COLORS[gate],
                    edgecolor="white", alpha=0.85)
            ax.axvline(mean, color="black", linewidth=1.6, linestyle="--",
                       label=f"μ={mean:.4f}\nσ={std:.4f}")
            ax.set_title(f"{gate.upper()} — {NOISE_LABELS[ch]}", fontsize=9)
            ax.set_xlabel("Noise value")
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)

    fig.suptitle("Noise distributions by gate type", fontsize=13)
    fig.tight_layout()
    return _save_and_return(fig, filepath)


# ---------------------------------------------------------------------------
# Plot 3 — Noise breakdown by qubit
# ---------------------------------------------------------------------------

def plot_noise_by_qubit(  # pragma: no cover  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    actions_data: Dict[str, Any],
    skip_identity: bool = True,
    skip_zero: bool = True,
    bins: int = 30,
    figsize: Optional[tuple] = None,
    filepath: Optional[str] = None,
) -> plt.Figure:
    """Plot noise distributions split by qubit index.

    Each row = noise channel; each column = qubit.

    Args:
        actions_data: Output of :func:`collect_actions`.
        skip_identity: Exclude identity-gate positions.
        skip_zero: Exclude positions where the agent output zero.
        bins: Number of histogram bins.
        figsize: Figure size; auto-computed if ``None``.
        filepath: Save path for the figure (optional).

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    n_qubits = actions_data["n_qubits"]
    n_rows = len(NOISE_CHANNELS)
    n_cols = n_qubits
    qubit_colors = ["#0bb4ff", "#e60049", "#50e991", "#9b19f5",
                    "#f46a9b", "#ffa300", "#b3d4ff"]

    if figsize is None:
        figsize = (4.5 * n_cols, 3.5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for r, ch in enumerate(NOISE_CHANNELS):
        for q in range(n_qubits):
            ax = axes[r][q]
            vals = _flat_values(actions_data, ch, qubit_filter=q,
                                skip_identity=skip_identity, skip_zero=skip_zero)
            if vals.size == 0:
                ax.set_visible(False)
                continue
            mean, std = vals.mean(), vals.std()
            color = qubit_colors[q % len(qubit_colors)]
            ax.hist(vals, bins=bins, color=color, edgecolor="white", alpha=0.85)
            ax.axvline(mean, color="black", linewidth=1.6, linestyle="--",
                       label=f"μ={mean:.4f}\nσ={std:.4f}")
            ax.set_title(f"Qubit {q} — {NOISE_LABELS[ch]}", fontsize=9)
            ax.set_xlabel("Noise value")
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)

    fig.suptitle("Noise distributions by qubit", fontsize=13)
    fig.tight_layout()
    return _save_and_return(fig, filepath)


# ---------------------------------------------------------------------------
# Plot 4 — Spatial noise profile (noise vs moment position)
# ---------------------------------------------------------------------------

def plot_spatial_noise(  # pragma: no cover  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    actions_data: Dict[str, Any],
    skip_identity: bool = True,
    skip_zero: bool = True,
    figsize: Optional[tuple] = None,
    filepath: Optional[str] = None,
) -> plt.Figure:
    """Plot mean noise ± std as a function of circuit moment position.

    This reveals whether the agent applies systematically more (or less)
    noise at the beginning / end of a circuit.

    Args:
        actions_data: Output of :func:`collect_actions`.
        skip_identity: Exclude identity positions from statistics.
        skip_zero: Exclude positions where the agent output zero.
        figsize: Figure size; defaults to ``(14, 4)``.
        filepath: Save path for the figure (optional).

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    n_moments = actions_data["n_moments"]
    moments = np.arange(n_moments)

    if figsize is None:
        figsize = (14, 4)

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    gate_type = actions_data["gate_type"]  # (n_c, n_m, n_q)

    for ax, ch in zip(axes, NOISE_CHANNELS):
        vals = actions_data[ch]  # (n_circuits, n_moments, n_qubits)
        means = np.zeros(n_moments)
        stds  = np.zeros(n_moments)

        for m in range(n_moments):
            slice_m = vals[:, m, :]        # (n_circuits, n_qubits)
            mask = np.ones_like(slice_m, dtype=bool)
            if skip_identity:
                gt_m = gate_type[:, m, :]  # (n_circuits, n_qubits)
                mask &= gt_m != "id"
            if skip_zero:
                mask &= slice_m != 0.0
            data = slice_m[mask]

            if data.size > 0:
                means[m] = data.mean()
                stds[m]  = data.std()

        color = NOISE_COLORS[ch]
        ax.plot(moments, means, color=color, linewidth=2)
        ax.fill_between(moments, means - stds, means + stds,
                        alpha=0.25, color=color)
        ax.set_xlabel("Moment index")
        ax.set_ylabel("Noise value")
        ax.set_title(NOISE_LABELS[ch])
        ax.grid(True, alpha=0.3)

    fig.suptitle("Noise profile along circuit depth", fontsize=13)
    fig.tight_layout()
    return _save_and_return(fig, filepath)


# ---------------------------------------------------------------------------
# Plot 5 — Per-qubit spatial noise profile
# ---------------------------------------------------------------------------

def plot_spatial_noise_per_qubit(  # pragma: no cover  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    actions_data: Dict[str, Any],
    channel: str = "depol",
    skip_identity: bool = True,
    skip_zero: bool = True,
    figsize: Optional[tuple] = None,
    filepath: Optional[str] = None,
) -> plt.Figure:
    """Plot mean noise vs circuit position for each qubit on a single axis.

    Args:
        actions_data: Output of :func:`collect_actions`.
        channel: One of ``"epsilon_x"``, ``"epsilon_z"``, ``"reset"``,
            ``"depol"``.
        skip_identity: Exclude identity positions.
        skip_zero: Exclude positions where the agent output zero.
        figsize: Figure size; defaults to ``(8, 4)``.
        filepath: Save path for the figure (optional).

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    if channel not in NOISE_CHANNELS:
        raise ValueError(f"channel must be one of {NOISE_CHANNELS}")

    n_moments = actions_data["n_moments"]
    n_qubits  = actions_data["n_qubits"]
    moments   = np.arange(n_moments)
    gate_type = actions_data["gate_type"]
    vals      = actions_data[channel]

    qubit_colors = ["#0bb4ff", "#e60049", "#50e991", "#9b19f5",
                    "#f46a9b", "#ffa300", "#b3d4ff"]

    if figsize is None:
        figsize = (8, 4)

    fig, ax = plt.subplots(figsize=figsize)

    for q in range(n_qubits):
        means = np.zeros(n_moments)
        stds  = np.zeros(n_moments)
        for m in range(n_moments):
            data_m = vals[:, m, q]
            mask = np.ones_like(data_m, dtype=bool)
            if skip_identity:
                mask &= gate_type[:, m, q] != "id"
            if skip_zero:
                mask &= data_m != 0.0
            data = data_m[mask]
            if data.size > 0:
                means[m] = data.mean()
                stds[m]  = data.std()
        color = qubit_colors[q % len(qubit_colors)]
        ax.plot(moments, means, color=color, linewidth=2, label=f"Qubit {q}")
        ax.fill_between(moments, means - stds, means + stds,
                        alpha=0.15, color=color)

    ax.set_xlabel("Moment index")
    ax.set_ylabel("Noise value")
    ax.set_title(f"{NOISE_LABELS[channel]} — per qubit")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_and_return(fig, filepath)


# ---------------------------------------------------------------------------
# Plot 6 — Noise correlation scatter matrix
# ---------------------------------------------------------------------------

def plot_noise_correlation(  # pragma: no cover  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    actions_data: Dict[str, Any],
    skip_identity: bool = True,
    skip_zero: bool = True,
    max_points: int = 3000,
    figsize: Optional[tuple] = None,
    filepath: Optional[str] = None,
) -> plt.Figure:
    """Pairwise scatter plot of all noise channels.

    Shows whether the agent correlates multiple noise types when assigning
    noise to a gate position.

    Args:
        actions_data: Output of :func:`collect_actions`.
        skip_identity: Exclude identity positions.
        skip_zero: Exclude positions where all channels are zero.  A shared
            mask is used so all channels remain aligned for scatter plots.
        max_points: Subsample to at most this many scatter points (for speed).
        figsize: Figure size; defaults to ``(12, 12)``.
        filepath: Save path for the figure (optional).

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    if figsize is None:
        figsize = (12, 12)

    n = len(NOISE_CHANNELS)
    shape = (
        actions_data["n_circuits"],
        actions_data["n_moments"],
        actions_data["n_qubits"],
    )
    combined_mask = np.ones(shape, dtype=bool)
    if skip_identity:
        combined_mask &= actions_data["gate_type"] != "id"
    if skip_zero:
        any_nonzero = np.zeros(shape, dtype=bool)
        for _ch in NOISE_CHANNELS:
            any_nonzero |= actions_data[_ch] != 0.0
        combined_mask &= any_nonzero

    data_flat: Dict[str, np.ndarray] = {}
    for ch in NOISE_CHANNELS:
        data_flat[ch] = actions_data[ch][combined_mask].ravel()

    # Subsample
    n_pts = len(data_flat[NOISE_CHANNELS[0]])
    if n_pts > max_points:
        idx = np.random.choice(n_pts, max_points, replace=False)
        data_flat = {ch: v[idx] for ch, v in data_flat.items()}

    fig, axes = plt.subplots(n, n, figsize=figsize)

    for r, ch_y in enumerate(NOISE_CHANNELS):
        for c, ch_x in enumerate(NOISE_CHANNELS):
            ax = axes[r][c]
            if r == c:
                # Diagonal: histogram
                ax.hist(data_flat[ch_x], bins=30,
                        color=NOISE_COLORS[ch_x], edgecolor="white", alpha=0.85)
                ax.set_xlabel(NOISE_LABELS[ch_x], fontsize=8)
            else:
                corr = np.corrcoef(data_flat[ch_x], data_flat[ch_y])[0, 1]
                ax.scatter(data_flat[ch_x], data_flat[ch_y],
                           s=6, alpha=0.3, color=NOISE_COLORS[ch_x])
                ax.set_xlabel(NOISE_LABELS[ch_x], fontsize=7)
                ax.set_ylabel(NOISE_LABELS[ch_y], fontsize=7)
                ax.set_title(f"r = {corr:.3f}", fontsize=9)
            ax.tick_params(labelsize=7)

    fig.suptitle("Pairwise noise-channel correlations", fontsize=13)
    fig.tight_layout()
    return _save_and_return(fig, filepath)


# ---------------------------------------------------------------------------
# Plot 7 — Mean noise per gate type, bar chart
# ---------------------------------------------------------------------------

def plot_mean_noise_per_gate(  # pragma: no cover  # pylint: disable=too-many-locals
    actions_data: Dict[str, Any],
    skip_zero: bool = True,
    figsize: Optional[tuple] = None,
    filepath: Optional[str] = None,
) -> plt.Figure:
    """Bar chart: mean ± std of each noise channel grouped by gate type.

    This is the most compact view to compare how much noise the agent
    assigns to each gate type.

    Args:
        actions_data: Output of :func:`collect_actions`.
        skip_zero: Exclude positions where the agent output zero.
        figsize: Figure size; defaults to ``(12, 5)``.
        filepath: Save path for the figure (optional).

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    present_gates = [g for g in ("rx", "rz", "cz")
                     if np.any(actions_data["gate_type"] == g)]
    if not present_gates:
        raise ValueError("No named gates found in actions_data.")

    if figsize is None:
        figsize = (12, 5)

    fig, axes = plt.subplots(1, len(NOISE_CHANNELS), figsize=figsize)

    for ax, ch in zip(axes, NOISE_CHANNELS):
        means, stds = [], []
        for gate in present_gates:
            v = _flat_values(actions_data, ch, gate_filter=gate,
                             skip_identity=True, skip_zero=skip_zero)
            means.append(v.mean() if v.size > 0 else 0.0)
            stds.append(v.std()  if v.size > 0 else 0.0)

        x = np.arange(len(present_gates))
        colors = [GATE_COLORS[g] for g in present_gates]
        ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([g.upper() for g in present_gates])
        ax.set_ylabel("Mean noise value")
        ax.set_title(NOISE_LABELS[ch])
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Mean noise per gate type (± std)", fontsize=13)
    fig.tight_layout()
    return _save_and_return(fig, filepath)
