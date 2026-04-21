"""04_evaluate_circuits.py — Evaluate the agent on random and structured circuits.

Reads the experiment configuration from
``experiments/experiment_configurations/<folder_name>.json``, loads the
trained RL agent and the held-out evaluation dataset, and runs:

  1. Dataset-level evaluation (random circuits from eval_dataset.npz).
  2. Visualisation of one randomly-sampled circuit (shots, DM heatmap, metrics).
  3. Structured circuits (Grover search + QFT) for multi-qubit experiments.

Outputs:

  results/benchmarking/eval_results.npz                   — per-circuit metrics
  results/benchmarking/images/eval_fidelity.png           — fidelity histogram
  results/benchmarking/images/eval_trace.png              — trace-distance histogram
  results/benchmarking/images/random_shots.png            — random circuit shots
  results/benchmarking/images/random_dm_heatmap.png       — random circuit DM heatmap
  results/benchmarking/images/random_circuit_metrics.png  — random circuit metrics
  results/benchmarking/images/grover_shots.png            — Grover shots (3-qubit only)
  results/benchmarking/images/grover_dm_heatmap.png       — Grover DM heatmap (3-qubit only)
  results/benchmarking/images/grover_circuit_metrics.png  — Grover metrics (3-qubit only)
  results/benchmarking/images/grover_results.txt          — Grover summary (3-qubit only)
  results/benchmarking/images/qft_shots.png               — QFT shots (3-qubit only)
  results/benchmarking/images/qft_dm_heatmap.png          — QFT DM heatmap (3-qubit only)
  results/benchmarking/images/qft_circuit_metrics.png     — QFT metrics (3-qubit only)
  results/benchmarking/images/qft_results.txt             — QFT summary (3-qubit only)

Run from the project root or directly from the experiment folder::

    python experiments/1qubit/04_evaluate_circuits.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import load_config, build_noise_model, build_encoder, load_agent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rlnoise.benchmarking import evaluate_on_dataset, evaluate_circuit, load_rb_fit
from rlnoise.circuit_generator import CircuitGenerator
from rlnoise.dataset import CircuitDataset
from rlnoise.visualization import (
    plot_shots,
    plot_density_matrix_heatmap,
    plot_circuit_metrics,
)


def _save_circuit_plots(results: dict, name: str, images_dir: Path) -> None:
    """Save shots, DM heatmap, and circuit-metrics plots for a single circuit."""
    fig = plot_shots(
        results,
        show_truth=True,
        show_rl=True,
        show_no_noise=True,
        show_mms=True,
        title=f"{name} — Basis-state probabilities",
        filepath=str(images_dir / f"{name.lower()}_shots.png"),
    )
    plt.close(fig)

    fig = plot_density_matrix_heatmap(
        results,
        show_rl=True,
        show_no_noise=True,
        show_mms=True,
        title=f"{name} — |ρ_truth − ρ_model|",
        filepath=str(images_dir / f"{name.lower()}_dm_heatmap.png"),
    )
    plt.close(fig)

    fig = plot_circuit_metrics(
        results,
        title=f"{name} — Metrics",
        filepath=str(images_dir / f"{name.lower()}_circuit_metrics.png"),
    )
    plt.close(fig)


def _save_metrics_histogram(values, xlabel, title, filepath):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=30, color="#0bb4ff", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(values), color="#e60049", linewidth=2,
               label=f"Mean = {np.mean(values):.4f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def main() -> None:
    cfg, dirs = load_config(__file__)

    noise_model = build_noise_model(cfg)
    encoder = build_encoder(cfg)
    agent = load_agent(cfg, dirs)

    # ── Load RB fit parameters if available ───────────────────────────────────
    rb_fit_path = dirs["results"] / "rb" / "rb_fit.json"
    lambda_rb: float | None = None
    if rb_fit_path.exists():
        _, lambda_rb = load_rb_fit(str(rb_fit_path))
        print(f"Loaded RB fit: lambda_rb = {lambda_rb:.6f}")
    else:
        print(f"[warn] No RB fit found at {rb_fit_path} \u2014 RB model will be skipped.")

    images_dir = dirs["results"] / "benchmarking" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Evaluation dataset ─────────────────────────────────────────────────
    eval_path = dirs["results"] / "dataset" / "eval_dataset.npz"
    if not eval_path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found at {eval_path}\n"
            "Run 01_generate_dataset.py first."
        )
    eval_dataset = CircuitDataset.load(str(eval_path))
    circuits = eval_dataset.circuits
    labels   = eval_dataset.labels

    print(f"\n=== Evaluating on {len(circuits)} random circuits ===")
    eval_results = evaluate_on_dataset(
        rl_agent=agent,
        circuits=circuits,
        labels=labels,
        verbose=True,
    )

    print(f"\nMean fidelity       : {eval_results['mean_fidelity']:.4f} ± {eval_results['std_fidelity']:.4f}")
    print(f"Mean trace distance : {eval_results['mean_trace_distance']:.4f} ± {eval_results['std_trace_distance']:.4f}")
    print(f"Mean MSE            : {eval_results['mean_mse']:.6f} ± {eval_results['std_mse']:.6f}")

    # Save arrays
    benchmarking_dir = dirs["results"] / "benchmarking"
    benchmarking_dir.mkdir(parents=True, exist_ok=True)
    npz_path = benchmarking_dir / "eval_results.npz"
    np.savez(
        str(npz_path),
        fidelity=np.array([r["fidelity"] for r in eval_results["per_circuit"]]),
        trace_distance=np.array([r["trace_distance"] for r in eval_results["per_circuit"]]),
        mse=np.array([r["mse"] for r in eval_results["per_circuit"]]),
    )
    print(f"\nEval results saved → {npz_path}")

    # Histograms
    fidelities = np.array([r["fidelity"] for r in eval_results["per_circuit"]])
    traces     = np.array([r["trace_distance"] for r in eval_results["per_circuit"]])

    _save_metrics_histogram(
        fidelities, "Fidelity",
        f"Fidelity distribution — {dirs['exp'].name}",
        images_dir / "eval_fidelity.png",
    )
    _save_metrics_histogram(
        traces, "Trace distance",
        f"Trace distance distribution — {dirs['exp'].name}",
        images_dir / "eval_trace.png",
    )
    print(f"Histograms saved → {images_dir}")

    # ── 2. Random circuit visualisation ────────────────────────────────────
    print("\n=== Evaluating a random circuit for visualisation ===")
    circuit_gen = CircuitGenerator(cfg.dataset.model_copy(update={"clifford": False}))
    random_circ = circuit_gen.generate_random_circuit()
    random_results = evaluate_circuit(
        circuit=random_circ,
        encoder=encoder,
        rl_agent=agent,
        noise_model=noise_model,
        lambda_rb=lambda_rb,
        evaluate_mms=True,
        evaluate_no_noise=True,
    )
    _save_circuit_plots(random_results, "random", images_dir)
    print(f"Random circuit plots saved → {images_dir}")

    # ── 3. Structured circuits (3-qubit only) ─────────────────────────────────
    if cfg.dataset.qubits == 3:
        from rlnoise.circuit_generator import grover_circuit, qft_circuit

        for name, circ_fn in [("Grover", grover_circuit), ("QFT", qft_circuit)]:
            print(f"\n=== Evaluating on {name} circuit ===")
            circ = circ_fn()
            sc_results = evaluate_circuit(
                circuit=circ,
                encoder=encoder,
                rl_agent=agent,
                noise_model=noise_model,
                lambda_rb=lambda_rb,
                evaluate_mms=True,
                evaluate_no_noise=True,
            )
            summary_lines = [
                f"{name} circuit evaluation",
                "-" * 40,
                f"  Gates             : {sc_results['n_gates']}",
                f"  Qubits            : {sc_results['n_qubits']}",
            ]
            for model_key, mvals in sc_results["metrics"].items():
                labels_map = {"rl": "RL model", "no_noise": "No noise", "mms": "MMS"}
                lbl = labels_map.get(model_key, model_key)
                summary_lines.append(
                    f"  [{lbl}]  fidelity={mvals['fidelity']:.4f}  "
                    f"trace={mvals['trace']:.4f}  mse={mvals['mse']:.6f}"
                )
            summary_text = "\n".join(summary_lines)
            print(summary_text)

            txt_path = images_dir / f"{name.lower()}_results.txt"
            txt_path.write_text(summary_text, encoding="utf-8")
            print(f"Summary saved → {txt_path}")

            _save_circuit_plots(sc_results, name, images_dir)
            print(f"{name} plots saved → {images_dir}")

    print("\nCircuit evaluation complete.")


if __name__ == "__main__":
    main()
