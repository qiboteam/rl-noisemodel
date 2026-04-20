"""02_train.py — Train the RL agent.

Reads the experiment configuration from
``experiments/experiment_configurations/<folder_name>.json``,
loads the training dataset produced by 01_generate_dataset.py, and trains a
PPO agent.  Outputs:

  results/model.zip           — best model weights
  results/training_history.npz — full training history
  results/training_dashboard.png — training-progress plot

Run from the project root or directly from the experiment folder::

    python experiments/1qubit/02_train.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import load_config, build_env, build_agent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rlnoise.dataset import CircuitDataset
from rlnoise.visualization import plot_training_dashboard


def main() -> None:
    cfg, dirs = load_config(__file__)

    # ── Load training dataset ─────────────────────────────────────────────────
    dataset_path = dirs["results"] / "dataset.npz"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {dataset_path}\n"
            "Run 01_generate_dataset.py first."
        )
    dataset = CircuitDataset.load(str(dataset_path))
    print(f"Loaded training dataset: {dataset}")

    # ── Build environment and agent ───────────────────────────────────────────
    env   = build_env(cfg, dataset)
    agent = build_agent(cfg, env)
    print(env)
    print(agent)

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n=== Training for {cfg.total_timesteps:,} timesteps ===")
    history = agent.train(
        total_timesteps=cfg.total_timesteps,
        check_freq=cfg.check_freq,
        save_best=True,
        progress_bar=True,
        verbose=True,
        save_path=str(dirs["results"] / "training" / "model"),
        history_path=str(dirs["results"] / "training" / "training_history"),
    )

    # ── Save training dashboard ───────────────────────────────────────────────
    fig = plot_training_dashboard(
        history,
        title=f"Training: {dirs['exp'].name}",
        filepath=str(dirs["results"] / "training" / "training_dashboard.png"),
        show_reward=True,
        show_trace_distance=True,
        show_fidelity=True,
    )
    plt.close(fig)
    print(f"\nTraining complete.")
    print(f"  Model   → {dirs['results'] / 'training' / 'model.zip'}")
    print(f"  History → {dirs['results'] / 'training' / 'training_history.npz'}")
    print(f"  Plot    → {dirs['results'] / 'training' / 'training_dashboard.png'}")


if __name__ == "__main__":
    main()
