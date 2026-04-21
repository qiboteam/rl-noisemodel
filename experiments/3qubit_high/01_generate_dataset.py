"""01_generate_dataset.py — Generate training and evaluation datasets.

Reads the experiment configuration from
``experiments/experiment_configurations/<folder_name>.json`` and writes:

  results/dataset.npz        — training dataset
  results/eval_dataset.npz   — held-out evaluation dataset (non-Clifford)

Run from the project root or directly from the experiment folder::

    python experiments/1qubit/01_generate_dataset.py
"""

import sys
from pathlib import Path

# Make _common importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import load_config, build_noise_model

from rlnoise.config import DatasetConfig
from rlnoise.dataset import DatasetGenerator


def main() -> None:
    cfg, dirs = load_config(__file__)

    noise_model = build_noise_model(cfg)

    # ── Training dataset ──────────────────────────────────────────────────────
    print("\n=== Generating training dataset ===")
    print(cfg.dataset)
    generator = DatasetGenerator(cfg.dataset, cfg.noise)
    dataset = generator.generate(verbose=True)
    save_path = dirs["results"] / "dataset" / "dataset.npz"
    dataset.save(str(save_path))
    print(f"Training dataset saved → {save_path}")

    # ── Evaluation dataset ────────────────────────────────────────────────────
    print("\n=== Generating evaluation dataset ===")
    eval_dataset_cfg = DatasetConfig(
        n_circuits=cfg.eval_n_circuits,
        moments=cfg.eval_depth,
        qubits=cfg.dataset.qubits,
        primitive_gates=cfg.dataset.primitive_gates,
        clifford=cfg.eval_clifford,
        mixed=False,
    )
    print(eval_dataset_cfg)
    eval_generator = DatasetGenerator(eval_dataset_cfg, cfg.noise)
    eval_dataset = eval_generator.generate(verbose=True)
    eval_save_path = dirs["results"] / "dataset" / "eval_dataset.npz"
    eval_dataset.save(str(eval_save_path))
    print(f"Evaluation dataset saved → {eval_save_path}")

    print("\nDataset generation complete.")


if __name__ == "__main__":
    main()
