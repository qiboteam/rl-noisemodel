"""Shared helpers for all experiment scripts.

Each experiment folder (1qubit/, 3qubit_high/, 3qubit_low/) contains five
scripts that all follow the same pattern:

  1. Resolve the configuration JSON and results directory from the folder name.
  2. Load ExperimentConfig from JSON.
  3. Build the objects needed for that script.
  4. Run the analysis / training / evaluation.
  5. Save all outputs (model, history, plots, npz) to the results directory.

This module provides the shared boilerplate so each script stays focused
on its own task.

Usage from any experiment script::

    from _common import load_config, make_dirs, build_env, build_agent, load_agent

    cfg, dirs = load_config()   # auto-detects folder from __file__
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Ensure the package is importable when running scripts directly
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # …/rl-noisemodel
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import qibo
qibo.set_backend("numpy")

from rlnoise.config import ExperimentConfig
from rlnoise.circuit_encoder import CircuitEncoder
from rlnoise.dataset import CircuitDataset, DatasetGenerator
from rlnoise.gym_env import QuantumCircuitEnv
from rlnoise.noise_model import QuantumNoiseModel
from rlnoise.rl_agent import RLAgent


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def load_config(caller_file: str | None = None) -> Tuple[ExperimentConfig, dict]:
    """Load ExperimentConfig from the JSON that matches this experiment folder.

    The function resolves the experiment name from the *parent directory* of the
    calling script, looks up the JSON in
    ``experiments/experiment_configurations/<name>.json``, and returns both the
    config object and a ``dirs`` dict of ready-to-use :class:`~pathlib.Path`
    objects.

    Args:
        caller_file: Pass ``__file__`` from the calling script.  Falls back to
            inspecting the call stack when omitted.

    Returns:
        ``(cfg, dirs)`` where *dirs* has keys
        ``"exp"``, ``"results"``.
    """
    if caller_file is None:
        import inspect
        frame = inspect.stack()[1]
        caller_file = frame.filename

    exp_dir = Path(caller_file).resolve().parent          # e.g. …/experiments/1qubit
    config_path = exp_dir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Expected a 'config.json' in the experiment folder {exp_dir}"
        )

    cfg = ExperimentConfig.from_json_file(str(config_path))

    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    dirs = {
        "exp":     exp_dir,
        "results": results_dir,
    }
    print(f"[config]  {config_path}")
    print(f"[results] {results_dir}")
    return cfg, dirs


def build_noise_model(cfg: ExperimentConfig) -> QuantumNoiseModel:
    """Construct a :class:`QuantumNoiseModel` from *cfg*."""
    return QuantumNoiseModel(cfg.noise, qubits=cfg.dataset.qubits)


def build_encoder(cfg: ExperimentConfig) -> CircuitEncoder:
    """Construct a :class:`CircuitEncoder` from *cfg*."""
    return CircuitEncoder(primitive_gates=cfg.dataset.primitive_gates)


def build_env(cfg: ExperimentConfig, dataset: CircuitDataset) -> QuantumCircuitEnv:
    """Construct a :class:`QuantumCircuitEnv` from *cfg* and a loaded dataset."""
    if cfg.gym_env is None:
        raise ValueError("ExperimentConfig.gym_env is required to build an environment.")
    if cfg.reward is None:
        raise ValueError("ExperimentConfig.reward is required to build an environment.")
    return QuantumCircuitEnv(
        dataset=dataset,
        encoder=build_encoder(cfg),
        env_config=cfg.gym_env,
        reward_config=cfg.reward,
    )


def build_agent(cfg: ExperimentConfig, env: QuantumCircuitEnv,
                model_path: str | None = None) -> RLAgent:
    """Construct an :class:`RLAgent` from *cfg*, optionally loading weights.

    Args:
        cfg: Experiment configuration.
        env: Pre-built environment.
        model_path: Path to a saved ``.zip`` model.  If ``None`` a fresh agent
            is created.
    """
    if cfg.agent is None:
        raise ValueError("ExperimentConfig.agent is required to build an agent.")
    return RLAgent(
        env=env,
        agent_config=cfg.agent,
        model_path=model_path,
    )


def load_agent(cfg: ExperimentConfig, dirs: dict,
               dataset: CircuitDataset | None = None) -> RLAgent:
    """Load the best saved agent from the results directory.

    Args:
        cfg: Experiment configuration.
        dirs: The *dirs* dict returned by :func:`load_config`.
        dataset: If provided, used to build the environment.  If ``None`` the
            training dataset is loaded from ``results/dataset.npz``.
    """
    if dataset is None:
        dataset_path = dirs["results"] / "dataset" / "dataset.npz"
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Training dataset not found at {dataset_path}.  "
                "Run 01_generate_dataset.py first."
            )
        dataset = CircuitDataset.load(str(dataset_path))

    env = build_env(cfg, dataset)
    model_path = str(dirs["results"] / "training" / "model")
    if not Path(model_path + ".zip").exists():
        raise FileNotFoundError(
            f"Saved model not found at {model_path}.zip.  "
            "Run 02_train.py first."
        )
    return build_agent(cfg, env, model_path=model_path)
