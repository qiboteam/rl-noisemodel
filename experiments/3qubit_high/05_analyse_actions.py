"""05_analyse_actions.py — Study the actions the agent applies to circuits.

Reads the experiment configuration from
``experiments/experiment_configurations/<folder_name>.json``, loads the
trained RL agent and the held-out evaluation dataset, and runs all available
analysis plots from :mod:`rlnoise.analysis`.

For 3-qubit experiments the analysis is also run on the Grover search and
QFT structured circuits.

Outputs (all in results/images/analysis/)::

  noise_distributions.png
  noise_by_gate.png
  noise_by_qubit.png
  spatial_noise.png
  spatial_noise_per_qubit.png
  noise_correlation.png
  mean_noise_per_gate.png
  grover_noise_distributions.png  (3-qubit only)
  grover_noise_by_gate.png        (3-qubit only)
  qft_noise_distributions.png     (3-qubit only)
  qft_noise_by_gate.png           (3-qubit only)

Run from the project root or directly from the experiment folder::

    python experiments/1qubit/05_analyse_actions.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import load_config, build_encoder, load_agent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rlnoise.analysis import (
    collect_actions,
    noise_summary,
    plot_noise_distributions,
    plot_noise_by_gate,
    plot_noise_by_qubit,
    plot_spatial_noise,
    plot_spatial_noise_per_qubit,
    plot_noise_correlation,
    plot_mean_noise_per_gate,
)
from rlnoise.dataset import CircuitDataset


def _run_all_plots(data: dict, analysis_dir: Path, prefix: str = "") -> None:
    """Save all analysis plots to *analysis_dir* with optional filename prefix."""
    analysis_dir.mkdir(parents=True, exist_ok=True)

    def _fp(name: str) -> str:
        return str(analysis_dir / f"{prefix}{name}")

    fig = plot_noise_distributions(data, filepath=_fp("noise_distributions.png"))
    plt.close(fig)

    fig = plot_noise_by_gate(data, filepath=_fp("noise_by_gate.png"))
    plt.close(fig)

    fig = plot_noise_by_qubit(data, filepath=_fp("noise_by_qubit.png"))
    plt.close(fig)

    fig = plot_spatial_noise(data, filepath=_fp("spatial_noise.png"))
    plt.close(fig)

    fig = plot_spatial_noise_per_qubit(data, filepath=_fp("spatial_noise_per_qubit.png"))
    plt.close(fig)

    fig = plot_noise_correlation(data, filepath=_fp("noise_correlation.png"))
    plt.close(fig)

    fig = plot_mean_noise_per_gate(data, filepath=_fp("mean_noise_per_gate.png"))
    plt.close(fig)


def main() -> None:
    cfg, dirs = load_config(__file__)

    encoder = build_encoder(cfg)
    agent   = load_agent(cfg, dirs)

    analysis_dir = dirs["results"] / "analysis" / "images"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Random evaluation dataset ─────────────────────────────────────────
    eval_path = dirs["results"] / "dataset" / "eval_dataset.npz"
    if not eval_path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found at {eval_path}\n"
            "Run 01_generate_dataset.py first."
        )
    eval_dataset = CircuitDataset.load(str(eval_path))
    circuits = eval_dataset.circuits

    print(f"\n=== Collecting actions on {len(circuits)} random circuits ===")
    data = collect_actions(agent, circuits, verbose=True)

    print("\nNoise summary:")
    print(noise_summary(data))

    _run_all_plots(data, analysis_dir, prefix="")
    print(f"Analysis plots saved → {analysis_dir}")

    # ── 2. Structured circuits (3-qubit only) ─────────────────────────────────
    if cfg.dataset.qubits == 3:
        from rlnoise.circuit_generator import grover_circuit, qft_circuit

        for name, circ_fn in [("grover", grover_circuit), ("qft", qft_circuit)]:
            print(f"\n=== Collecting actions on {name.upper()} circuit ===")
            circ = circ_fn()
            circ_array = encoder.circuit_to_array(circ)
            # Wrap single circuit in an array shape (1, moments, qubits, enc_dim)
            circ_batch = circ_array[np.newaxis]
            sc_data = collect_actions(agent, circ_batch, verbose=False)
            print(noise_summary(sc_data))
            _run_all_plots(sc_data, analysis_dir, prefix=f"{name}_")
            print(f"  {name.upper()} plots saved with prefix '{name}_'")

    print("\nAction analysis complete.")


if __name__ == "__main__":
    main()
