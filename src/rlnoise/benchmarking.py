"""Benchmarking utilities for comparing noise models.

Provides tools to:
- Generate randomized-benchmarking (RB) circuit datasets at various depths.
- Fit the RB exponential decay to extract a depolarizing parameter.
- Evaluate noise models side-by-side (RL, RB, no-noise, MMS — all optional).
- Summarize results in formatted tables.
"""

from typing import Any, Dict, List, Optional, Tuple

import json
import numpy as np
from qibo.models import Circuit
from qibo.noise import DepolarizingError, NoiseModel
from qibo.quantum_info import fidelity as qibo_fidelity
from scipy.optimize import curve_fit

from rlnoise.circuit_encoder import CircuitEncoder
from rlnoise.circuit_generator import CircuitGenerator
from rlnoise.noise_model import QuantumNoiseModel
from rlnoise.reward import mse, trace_distance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def maximally_mixed_state(dim: int) -> np.ndarray:
    """Return the density matrix of the maximally mixed state.

    Args:
        dim: Hilbert space dimension (2^n_qubits).

    Returns:
        Density matrix of shape (dim, dim).
    """
    return np.eye(dim, dtype=complex) / dim


def _build_composite_circuit(circuit: Circuit) -> Circuit:
    """Return a new circuit equal to *circuit* followed by its own inverse.

    The composite circuit is noiseless; it should return |0> in the absence of
    noise and is used to measure the RB survival probability.

    Args:
        circuit: Noiseless Qibo circuit.

    Returns:
        New circuit of twice the depth.
    """
    n = circuit.nqubits
    composite = Circuit(n, density_matrix=True)
    for g in circuit.queue:
        composite.add(g)
    inv = circuit.invert()
    for g in inv.queue:
        composite.add(g)
    return composite


def _apply_rb_noise_model(circuit: Circuit, lambda_rb: float) -> Circuit:
    """Apply a uniform depolarizing channel with parameter ``1 - lambda_rb``.

    Args:
        circuit: Noiseless Qibo circuit.
        lambda_rb: RB decay constant (0 < lambda_rb <= 1).

    Returns:
        Circuit with depolarizing noise applied to every gate.
    """
    p = max(0.0, 1.0 - lambda_rb)
    nm = NoiseModel()
    nm.add(DepolarizingError(p))
    return nm.apply(circuit)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_rb_circuits(
    circuit_gen: CircuitGenerator,
    encoder: CircuitEncoder,  # pylint: disable=unused-argument
    noise_model: QuantumNoiseModel,
    depths: List[int],
    n_circuits_per_depth: int,
) -> List[Dict[str, Any]]:
    """Generate RB circuit datasets at multiple depths.

    For each depth, ``n_circuits_per_depth`` random Clifford circuits are
    generated and run through ``noise_model`` to obtain target density matrices.

    Args:
        circuit_gen: CircuitGenerator instance (its ``n_moments`` attribute is
            temporarily modified and then restored).
        encoder: CircuitEncoder used to convert circuits to arrays.
        noise_model: QuantumNoiseModel used to simulate the target noisy DMs.
        depths: List of circuit depths (number of moments per circuit).
        n_circuits_per_depth: Number of circuits to generate per depth.

    Returns:
        List of dicts, one per depth, each containing:
            ``depth``           – int, circuit depth.
            ``circuit_arrays``  – ndarray (n, moments, n_qubits, enc_dim).
            ``qibo_circuits``   – list of noiseless Qibo Circuit objects.
            ``labels``          – ndarray (n, dim, dim) target density matrices.
    """
    original_moments = circuit_gen.n_moments
    rb_data: List[Dict[str, Any]] = []

    for depth in depths:
        print(f"  Generating {n_circuits_per_depth} circuits of depth {depth}…")
        circuit_gen.n_moments = depth

        qibo_circuits = [
            circuit_gen.generate_random_circuit()
            for _ in range(n_circuits_per_depth)
        ]

        # Target DMs via noise model simulation
        labels = np.array(
            [noise_model.apply(c)().state() for c in qibo_circuits]  # type: ignore[union-attr]
        )

        circuit_arrays = np.array(
            [encoder.circuit_to_array(c) for c in qibo_circuits]
        )

        rb_data.append(
            {
                "depth": depth,
                "circuit_arrays": circuit_arrays,
                "qibo_circuits": qibo_circuits,
                "labels": labels,
            }
        )

    circuit_gen.n_moments = original_moments
    return rb_data


# ---------------------------------------------------------------------------
# RB fit
# ---------------------------------------------------------------------------

def fit_rb_decay(  # pylint: disable=too-many-locals
    rb_data: List[Dict[str, Any]],
    noise_model: QuantumNoiseModel,
) -> Tuple[float, float]:
    """Fit the RB exponential decay and return the depolarizing parameter.

    For each circuit C of depth *d*, the composite circuit ``C + C^{-1}`` is
    constructed, noise is applied, and the survival probability
    ``P(|0…0>)`` is read from the diagonal of the resulting density matrix.
    The average ``P`` per depth is then fitted to ``a * lambda^depth``.

    Args:
        rb_data: Output of :func:`generate_rb_circuits`.
        noise_model: QuantumNoiseModel used to simulate the noisy evolution.

    Returns:
        Tuple ``(a, lambda_rb)`` where ``lambda_rb`` is the decay constant
        (close to 1 for low noise).
    """
    depths_arr: List[float] = []
    avg_survival_arr: List[float] = []

    for entry in rb_data:
        depth = entry["depth"]
        survival: List[float] = []

        for c in entry["qibo_circuits"]:
            composite = _build_composite_circuit(c)
            noisy = noise_model.apply(composite)
            dm = noisy().state()  # type: ignore[union-attr]
            p0 = float(np.real(dm[0, 0]))
            survival.append(p0)

        depths_arr.append(float(depth))
        avg_survival_arr.append(float(np.mean(survival)))

    depths_np = np.array(depths_arr)
    survival_np = np.array(avg_survival_arr)

    def model_fn(d, a, lam):
        return a * np.power(lam, d)


    try:
        fit_result = curve_fit(
            model_fn,
            depths_np,
            survival_np,
            p0=[1.0, 0.9],
            maxfev=5000,
            bounds=([0.0, 0.0], [2.0, 1.0]),
        )
        popt = fit_result[0]
        a_fit, lambda_fit = float(popt[0]), float(popt[1])
    except RuntimeError:
        # Fallback: linear fit in log space
        log_s = np.log(np.clip(survival_np, 1e-10, None))
        slope = float(np.polyfit(depths_np, log_s, 1)[0])
        lambda_fit = float(np.exp(slope))
        a_fit = 1.0

    return a_fit, lambda_fit


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_benchmarks(  # pylint: disable=too-many-locals,too-many-branches,too-many-arguments,too-many-positional-arguments,unused-argument
    rb_data: List[Dict[str, Any]],
    encoder: CircuitEncoder,
    rl_agent,
    lambda_rb: Optional[float] = None,
    evaluate_mms: bool = True,
    evaluate_no_noise: bool = True,
) -> Dict[str, Any]:
    """Evaluate noise models at each circuit depth.

    The RL agent is always evaluated.  The other models are optional:

    * **RB model** — skipped when ``lambda_rb`` is ``None``.
    * **No-noise** — skipped when ``evaluate_no_noise=False``.
    * **MMS** — skipped when ``evaluate_mms=False``.

    For every circuit in ``rb_data`` each active model produces a density
    matrix that is compared to the target (noisy) label using fidelity, trace
    distance, and MSE.

    Args:
        rb_data: Output of :func:`generate_rb_circuits`.
        encoder: CircuitEncoder (used implicitly via rl_agent).
        rl_agent: Trained :class:`~rlnoise.rl_agent.RLAgent`.
        lambda_rb: Decay constant returned by :func:`fit_rb_decay`.  When
            ``None`` the RB baseline is skipped entirely.
        evaluate_mms: Whether to evaluate the maximally-mixed-state baseline.
        evaluate_no_noise: Whether to evaluate the noiseless simulation.

    Returns:
        Dictionary with keys:

        ``depths``
            List of circuit depths.

        ``rl``
            Always present.  Dict with ``fidelity``, ``fidelity_std``,
            ``trace``, ``trace_std``, ``mse``, ``mse_std``.

        ``rb``
            Present only when ``lambda_rb`` is not ``None``.

        ``no_noise``
            Present only when ``evaluate_no_noise=True``.

        ``mms``
            Present only when ``evaluate_mms=True``.
    """
    active_keys: List[str] = ["rl"]
    if lambda_rb is not None:
        active_keys.append("rb")
    if evaluate_no_noise:
        active_keys.append("no_noise")
    if evaluate_mms:
        active_keys.append("mms")

    depths_list: List[int] = []

    # Accumulate per-depth stats
    agg: Dict[str, Dict[str, List[float]]] = {
        k: {"fidelity": [], "fidelity_std": [], "trace": [], "trace_std": [],
            "mse": [], "mse_std": []}
        for k in active_keys
    }

    for entry in rb_data:
        depth = entry["depth"]
        circuit_arrays = entry["circuit_arrays"]
        qibo_circuits = entry["qibo_circuits"]
        labels = entry["labels"]
        dim = labels.shape[-1]

        print(f"  Evaluating depth {depth} ({len(qibo_circuits)} circuits)…")

        per_circ: Dict[str, Dict[str, List[float]]] = {
            k: {"fidelity": [], "trace": [], "mse": []} for k in active_keys
        }

        mms_dm = maximally_mixed_state(dim) if evaluate_mms else None

        for arr, qc, label in zip(circuit_arrays, qibo_circuits, labels):
            # RL
            rl_qc = rl_agent.apply_to_circuit(arr, return_qibo=True)
            dm = rl_qc().state()
            per_circ["rl"]["fidelity"].append(float(qibo_fidelity(label, dm)))
            per_circ["rl"]["trace"].append(float(trace_distance(label, dm)))
            per_circ["rl"]["mse"].append(float(mse(label, dm)))

            # RB (optional)
            if lambda_rb is not None:
                rb_qc = _apply_rb_noise_model(qc, lambda_rb)
                dm = rb_qc().state()  # type: ignore[union-attr]
                per_circ["rb"]["fidelity"].append(float(qibo_fidelity(label, dm)))
                per_circ["rb"]["trace"].append(float(trace_distance(label, dm)))
                per_circ["rb"]["mse"].append(float(mse(label, dm)))

            # No noise (optional)
            if evaluate_no_noise:
                dm = qc().state()
                per_circ["no_noise"]["fidelity"].append(float(qibo_fidelity(label, dm)))
                per_circ["no_noise"]["trace"].append(float(trace_distance(label, dm)))
                per_circ["no_noise"]["mse"].append(float(mse(label, dm)))

            # MMS (optional)
            if evaluate_mms:
                per_circ["mms"]["fidelity"].append(float(qibo_fidelity(label, mms_dm)))
                per_circ["mms"]["trace"].append(float(trace_distance(label, mms_dm)))
                per_circ["mms"]["mse"].append(float(mse(label, mms_dm)))

        depths_list.append(depth)
        for k in active_keys:
            for metric in ("fidelity", "trace", "mse"):
                vals = np.array(per_circ[k][metric])
                agg[k][metric].append(float(vals.mean()))
                agg[k][f"{metric}_std"].append(float(vals.std()))

    return {"depths": depths_list, **agg}


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

_MODEL_LABELS: Dict[str, str] = {
    "rl":       "RL model",
    "rb":       "Randomized benchmarking",
    "no_noise": "No noise",
    "mms":      "Maximally mixed state",
}


def summarize_benchmarks(results: Dict[str, Any]) -> str:
    """Return a formatted ASCII table summarising benchmark results per depth.

    Rows are grouped by depth.  For each depth every model present in
    *results* is printed with its mean fidelity, trace distance and MSE.

    Args:
        results: Dictionary returned by :func:`evaluate_benchmarks`.

    Returns:
        Multi-line string containing the formatted table (also printed to
        stdout).
    """
    model_keys = [k for k in ("rl", "rb", "no_noise", "mms") if k in results]
    depths = results["depths"]

    col_w = max(len(_MODEL_LABELS[k]) for k in model_keys)
    header = (
        f"{'Depth':>6}  {' Model':<{col_w}}  "
        f"{'Fidelity':>10}  {'Trace':>10}  {'MSE':>12}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for i, depth in enumerate(depths):
        for k in model_keys:
            lbl = _MODEL_LABELS[k]
            f_val  = results[k]["fidelity"][i]
            tr_val = results[k]["trace"][i]
            ms_val = results[k]["mse"][i]
            lines.append(
                f"{depth:>6}  {lbl:<{col_w}}  "
                f"{f_val:>10.4f}  {tr_val:>10.4f}  {ms_val:>12.6f}"
            )
        lines.append("")

    table = "\n".join(lines)
    print(table)
    return table


def summarize_rb_parameters(a: float, lambda_rb: float) -> str:
    """Return a formatted summary of RB fit parameters.

    Args:
        a: Amplitude from the RB decay fit.
        lambda_rb: Decay constant from the RB decay fit.

    Returns:
        Multi-line string containing the formatted summary (also printed to
        stdout).
    """
    p_eff = 1.0 - lambda_rb
    n_cliffords = 1.0 / p_eff if p_eff > 0 else float("inf")
    lines = [
        "RB Fit Parameters",
        "-" * 30,
        f"  a (amplitude)  : {a:.6f}",
        f"  λ (decay)      : {lambda_rb:.6f}",
        f"  p = 1 − λ      : {p_eff:.6f}",
        f"  1/p (Cliffords): {n_cliffords:.1f}",
    ]
    table = "\n".join(lines)
    print(table)
    return table


def save_rb_fit(a: float, lambda_rb: float, filepath: str) -> None:
    """Save RB fit parameters to a JSON file.

    Args:
        a: Amplitude from the RB decay fit.
        lambda_rb: Decay constant from the RB decay fit.
        filepath: Destination path.  The ``.json`` extension is added
            automatically when not present.
    """
    from pathlib import Path
    if not filepath.endswith(".json"):
        filepath = filepath + ".json"
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump({"a": float(a), "lambda_rb": float(lambda_rb)}, fh, indent=2)


def load_rb_fit(filepath: str) -> Tuple[float, float]:
    """Load RB fit parameters previously saved with :func:`save_rb_fit`.

    Args:
        filepath: Path to the ``.json`` file.  The ``.json`` extension is added
            automatically when not present.

    Returns:
        ``(a, lambda_rb)`` tuple of floats.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not filepath.endswith(".json"):
        filepath = filepath + ".json"
    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return float(data["a"]), float(data["lambda_rb"])


# ---------------------------------------------------------------------------
# Single-circuit evaluation
# ---------------------------------------------------------------------------

def evaluate_circuit(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches
    circuit: Circuit,
    encoder: CircuitEncoder,
    rl_agent,
    noise_model: QuantumNoiseModel,
    lambda_rb: Optional[float] = None,
    evaluate_mms: bool = True,
    evaluate_no_noise: bool = True,
) -> Dict[str, Any]:
    """Evaluate noise models on a single arbitrary circuit.

    The noisy ground-truth density matrix is produced by applying
    ``noise_model`` to *circuit*.  Each active model then predicts its own
    density matrix, which is compared against the ground truth with fidelity,
    trace distance, and MSE.

    Intended for structured benchmark circuits such as QFT or Grover, or for
    any user-supplied :class:`qibo.models.Circuit`.

    Args:
        circuit: Noiseless Qibo :class:`~qibo.models.Circuit`.
        encoder: :class:`~rlnoise.circuit_encoder.CircuitEncoder` used to
            convert the circuit to an array for the RL agent.
        rl_agent: Trained :class:`~rlnoise.rl_agent.RLAgent`.
        noise_model: :class:`~rlnoise.noise_model.QuantumNoiseModel` used to
            generate the ground-truth density matrix.
        lambda_rb: Decay constant from :func:`fit_rb_decay`.  When ``None``
            the RB uniform depolarising baseline is skipped.
        evaluate_mms: Whether to evaluate the maximally-mixed-state baseline.
        evaluate_no_noise: Whether to evaluate the noiseless simulation.

    Returns:
        Dictionary containing:

        ``n_qubits``
            Number of qubits in the circuit.

        ``n_gates``
            Total gate count (length of ``circuit.queue``).

        ``dm_truth``
            Ground-truth density matrix (``ndarray``).

        ``dm_rl``
            RL model density matrix (always present).

        ``dm_rb``
            RB model density matrix (present only when *lambda_rb* is given).

        ``dm_no_noise``
            Noiseless density matrix (present only when
            *evaluate_no_noise=True*).

        ``dm_mms``
            Maximally mixed state density matrix (present only when
            *evaluate_mms=True*).

        ``metrics``
            Nested dict ``{model_key: {'fidelity': float, 'trace': float,
            'mse': float}}``.  Keys follow the same convention as
            :func:`evaluate_benchmarks` (``rl``, ``rb``, ``no_noise``,
            ``mms``).
    """
    n_qubits = circuit.nqubits
    dim = 2 ** n_qubits

    # Ground truth via noise model
    dm_truth = noise_model.apply(circuit)().state()  # type: ignore[union-attr]

    result: Dict[str, Any] = {
        "n_qubits": n_qubits,
        "n_gates": len(circuit.queue),
        "dm_truth": dm_truth,
        "metrics": {},
    }

    # RL model (always active)
    circuit_array = encoder.circuit_to_array(circuit)
    rl_qc = rl_agent.apply_to_circuit(circuit_array, return_qibo=True)
    dm_rl = rl_qc().state()
    result["dm_rl"] = dm_rl
    result["metrics"]["rl"] = {
        "fidelity": float(qibo_fidelity(dm_truth, dm_rl)),
        "trace":    float(trace_distance(dm_truth, dm_rl)),
        "mse":      float(mse(dm_truth, dm_rl)),
    }

    # RB uniform depolarising baseline (optional)
    if lambda_rb is not None:
        rb_qc = _apply_rb_noise_model(circuit, lambda_rb)
        dm_rb = rb_qc().state()  # type: ignore[union-attr]
        result["dm_rb"] = dm_rb
        result["metrics"]["rb"] = {
            "fidelity": float(qibo_fidelity(dm_truth, dm_rb)),
            "trace":    float(trace_distance(dm_truth, dm_rb)),
            "mse":      float(mse(dm_truth, dm_rb)),
        }

    # Noiseless simulation (optional)
    if evaluate_no_noise:
        dm_no_noise = circuit().state()
        result["dm_no_noise"] = dm_no_noise
        result["metrics"]["no_noise"] = {
            "fidelity": float(qibo_fidelity(dm_truth, dm_no_noise)),
            "trace":    float(trace_distance(dm_truth, dm_no_noise)),
            "mse":      float(mse(dm_truth, dm_no_noise)),
        }

    # Maximally mixed state (optional)
    if evaluate_mms:
        dm_mms = maximally_mixed_state(dim)
        result["dm_mms"] = dm_mms
        result["metrics"]["mms"] = {
            "fidelity": float(qibo_fidelity(dm_truth, dm_mms)),
            "trace":    float(trace_distance(dm_truth, dm_mms)),
            "mse":      float(mse(dm_truth, dm_mms)),
        }

    return result


def evaluate_on_dataset(  # pylint: disable=too-many-locals
    rl_agent,
    circuits: np.ndarray,
    labels: np.ndarray,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate the RL agent on a held-out dataset of non-Clifford circuits.

    For each circuit the agent applies its learned noise policy, then the
    resulting density matrix is compared with the ground-truth label using
    fidelity, trace distance, and MSE.

    This reproduces the ``apply_eval_dataset`` evaluation from the original
    codebase (``old/src/rlnoise/rl_agent.py``).

    Args:
        rl_agent: Trained :class:`~rlnoise.rl_agent.RLAgent` instance.
        circuits: Array of encoded circuits, shape
            ``(n_circuits, n_moments, n_qubits, encoding_dim)``.
        labels: Ground-truth density matrices, shape
            ``(n_circuits, 2**n_qubits, 2**n_qubits)``.
        verbose: Print per-circuit progress and final averages.

    Returns:
        Dictionary with keys:

        - ``"per_circuit"`` — list of per-circuit dicts with ``fidelity``,
          ``trace_distance``, and ``mse`` keys.
        - ``"mean_fidelity"`` — float
        - ``"std_fidelity"`` — float
        - ``"mean_trace_distance"`` — float
        - ``"std_trace_distance"`` — float
        - ``"mean_mse"`` — float
        - ``"std_mse"`` — float
        - ``"n_circuits"`` — int
        - ``"predicted_dms"`` — list of predicted density matrices
    """
    n_circuits = len(circuits)
    per_circuit: List[Dict[str, float]] = []
    predicted_dms: List[np.ndarray] = []

    for i in range(n_circuits):
        if verbose and (i % max(1, n_circuits // 10) == 0):
            print(f"  Evaluating circuit {i + 1}/{n_circuits}…")

        noisy_qibo = rl_agent.apply_to_circuit(circuits[i], return_qibo=True)
        dm_pred = np.array(noisy_qibo().state())
        predicted_dms.append(dm_pred)

        dm_true = labels[i]
        fid = float(qibo_fidelity(dm_pred, dm_true))
        td = float(trace_distance(dm_pred, dm_true))
        mse_val = float(mse(dm_pred, dm_true))

        per_circuit.append({
            "fidelity": fid,
            "trace_distance": td,
            "mse": mse_val,
        })

    fidelities = np.array([r["fidelity"] for r in per_circuit])
    trace_dists = np.array([r["trace_distance"] for r in per_circuit])
    mses = np.array([r["mse"] for r in per_circuit])

    results = {
        "per_circuit": per_circuit,
        "mean_fidelity": float(fidelities.mean()),
        "std_fidelity": float(fidelities.std()),
        "mean_trace_distance": float(trace_dists.mean()),
        "std_trace_distance": float(trace_dists.std()),
        "mean_mse": float(mses.mean()),
        "std_mse": float(mses.std()),
        "n_circuits": n_circuits,
        "predicted_dms": predicted_dms,
    }

    if verbose:
        print(f"\n{'='*50}")
        print("  Evaluation Results (non-Clifford test set)")
        print(f"{'='*50}")
        print(f"  Circuits evaluated : {n_circuits}")
        print(f"  Fidelity           : {results['mean_fidelity']:.4f} ± {results['std_fidelity']:.4f}")
        print(f"  Trace distance     : {results['mean_trace_distance']:.4f} ± {results['std_trace_distance']:.4f}")
        print(f"  MSE                : {results['mean_mse']:.6f} ± {results['std_mse']:.6f}")
        print(f"{'='*50}")

    return results


def summarize_circuit_metrics(results: Dict[str, Any]) -> str:
    """Print and return a formatted table of single-circuit evaluation metrics.

    Args:
        results: Dictionary returned by :func:`evaluate_circuit`.

    Returns:
        Multi-line string containing the table (also printed to stdout).
    """
    metrics = results["metrics"]
    active_keys = [k for k in ("rl", "rb", "no_noise", "mms") if k in metrics]
    col_w = max(len(_MODEL_LABELS.get(k, k)) for k in active_keys)

    lines = [
        "Circuit Evaluation Results",
        f"  Qubits : {results['n_qubits']}",
        f"  Gates  : {results['n_gates']}",
        "-" * (col_w + 42),
        f"  {'Model':<{col_w}}  {'Fidelity':>10}  {'Trace':>10}  {'MSE':>12}",
        "-" * (col_w + 42),
    ]
    for k in active_keys:
        lbl = _MODEL_LABELS.get(k, k)
        m = metrics[k]
        lines.append(
            f"  {lbl:<{col_w}}  {m['fidelity']:>10.4f}  {m['trace']:>10.4f}  {m['mse']:>12.6f}"
        )
    table = "\n".join(lines)
    print(table)
    return table
