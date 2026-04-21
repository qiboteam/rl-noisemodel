"""03_evaluate_rb.py — Randomized Benchmarking evaluation.

Reads the experiment configuration from
``experiments/experiment_configurations/<folder_name>.json``, loads the
trained RL agent, and runs a complete RB benchmark comparison.  Outputs:

  results/rb_results.npz          — benchmark arrays per depth
  results/rb_decay.png            — RB decay fit curve
  results/rb_benchmarking.png     — fidelity / trace / MSE vs depth

Run from the project root or directly from the experiment folder::

    python experiments/1qubit/03_evaluate_rb.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import load_config, build_noise_model, build_encoder, load_agent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rlnoise.benchmarking import (
    generate_rb_circuits,
    fit_rb_decay,
    evaluate_benchmarks,
    summarize_benchmarks,
    summarize_rb_parameters,
    save_rb_fit,
)
from rlnoise.circuit_generator import CircuitGenerator
from rlnoise.visualization import plot_rb_decay, plot_benchmarking_results


def main() -> None:
    cfg, dirs = load_config(__file__)

    if cfg.rb is None:
        raise ValueError(
            "No 'rb' section found in the config JSON.  "
            "Add a 'rb' block with start/stop/step/n_circ."
        )

    noise_model = build_noise_model(cfg)
    encoder = build_encoder(cfg)
    agent = load_agent(cfg, dirs)

    # ── Build circuit generator ───────────────────────────────────────────────
    circuit_gen = CircuitGenerator(
        cfg.dataset.model_copy(update={"clifford": True})
    )

    depths = list(range(cfg.rb.start, cfg.rb.stop + 1, cfg.rb.step))
    print(f"\n=== Generating RB circuits at depths {depths} ===")
    rb_data = generate_rb_circuits(
        circuit_gen=circuit_gen,
        encoder=encoder,
        noise_model=noise_model,
        depths=depths,
        n_circuits_per_depth=cfg.rb.n_circ,
    )

    # ── Fit RB decay ──────────────────────────────────────────────────────────
    print("\n=== Fitting RB decay ===")
    a_fit, lambda_fit = fit_rb_decay(rb_data, noise_model)
    summarize_rb_parameters(a_fit, lambda_fit)

    # ── Evaluate all models ───────────────────────────────────────────────────
    print("\n=== Evaluating models vs RB ===")
    results = evaluate_benchmarks(
        rb_data=rb_data,
        encoder=encoder,
        rl_agent=agent,
        lambda_rb=lambda_fit,
        evaluate_mms=True,
        evaluate_no_noise=True,
    )
    summarize_benchmarks(results)

    # ── Save raw results ──────────────────────────────────────────────────────
    save_path = dirs["results"] / "rb" / "rb_results.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(save_path),
        depths=np.array(results["depths"]),
        a_fit=np.float64(a_fit),
        lambda_fit=np.float64(lambda_fit),
        rl_fidelity=np.array(results["rl"]["fidelity"]),
        rl_trace=np.array(results["rl"]["trace"]),
        rb_fidelity=np.array(results["rb"]["fidelity"]) if "rb" in results else np.array([]),
        rb_trace=np.array(results["rb"]["trace"]) if "rb" in results else np.array([]),
    )
    print(f"RB results saved → {save_path}")
    fit_path = dirs["results"] / "rb" / "rb_fit"
    save_rb_fit(a_fit, lambda_fit, str(fit_path))
    print(f"RB fit parameters saved → {fit_path}.json")
    # ── Plots ─────────────────────────────────────────────────────────────────
    images_dir = dirs["results"] / "rb" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    fig_decay = plot_rb_decay(
        rb_data=rb_data,
        a=a_fit,
        lambda_rb=lambda_fit,
        title=f"RB Decay — {dirs['exp'].name}",
        filepath=str(images_dir / "rb_decay.png"),
    )
    plt.close(fig_decay)

    fig_bench = plot_benchmarking_results(
        results=results,
        title=f"RB Benchmark — {dirs['exp'].name}",
        filepath=str(images_dir / "rb_benchmarking.png"),
    )
    plt.close(fig_bench)

    print(f"\nRB evaluation complete.")
    print(f"  Results → {save_path}")
    print(f"  Plots   → {images_dir}")


if __name__ == "__main__":
    main()
