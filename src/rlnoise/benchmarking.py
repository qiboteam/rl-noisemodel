"""Benchmarking utilities for comparing noise models.

Provides tools to:
- Generate randomized-benchmarking (RB) circuit datasets at various depths.
- Fit the RB exponential decay to extract a depolarizing parameter.
- Evaluate four noise models side-by-side:
    1. RL agent (trained model)
    2. RB model (fitted uniform depolarizing)
    3. No noise (noiseless simulation)
    4. Maximally mixed state (MMS)
"""

from typing import Any, Dict, List, Tuple

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
        popt, _ = curve_fit(
            model_fn,
            depths_np,
            survival_np,
            p0=[1.0, 0.9],
            maxfev=5000,
            bounds=([0.0, 0.0], [2.0, 1.0]),
        )
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

def evaluate_benchmarks(  # pylint: disable=too-many-locals,unused-argument
    rb_data: List[Dict[str, Any]],
    encoder: CircuitEncoder,
    rl_agent,
    lambda_rb: float,
) -> Dict[str, Any]:
    """Evaluate RL, RB, no-noise, and MMS models at each circuit depth.

    For every circuit in ``rb_data`` the four models produce density matrices
    that are compared to the target (noisy) label using fidelity, trace
    distance, and MSE.

    Args:
        rb_data: Output of :func:`generate_rb_circuits`.
        encoder: CircuitEncoder (used implicitly via rl_agent).
        rl_agent: Trained :class:`~rlnoise.rl_agent.RLAgent`.
        lambda_rb: Decay constant returned by :func:`fit_rb_decay`.

    Returns:
        Dictionary with keys:

        ``depths``
            List of circuit depths.

        ``rl``, ``rb``, ``no_noise``, ``mms``
            Each is a dict with keys ``fidelity``, ``fidelity_std``,
            ``trace``, ``trace_std``, ``mse``, ``mse_std`` — each a list
            of floats, one per depth.  Fidelity is higher-is-better;
            trace distance and MSE are lower-is-better.
    """
    model_keys = ["rl", "rb", "no_noise", "mms"]
    depths_list: List[int] = []

    # Accumulate per-depth stats
    agg: Dict[str, Dict[str, List[float]]] = {
        k: {"fidelity": [], "fidelity_std": [], "trace": [], "trace_std": [],
            "mse": [], "mse_std": []}
        for k in model_keys
    }

    for entry in rb_data:
        depth = entry["depth"]
        circuit_arrays = entry["circuit_arrays"]
        qibo_circuits = entry["qibo_circuits"]
        labels = entry["labels"]
        dim = labels.shape[-1]
        mms_dm = maximally_mixed_state(dim)

        print(f"  Evaluating depth {depth} ({len(qibo_circuits)} circuits)…")

        per_circ: Dict[str, Dict[str, List[float]]] = {
            k: {"fidelity": [], "trace": [], "mse": []} for k in model_keys
        }

        for arr, qc, label in zip(circuit_arrays, qibo_circuits, labels):
            # RL
            rl_qc = rl_agent.apply_to_circuit(arr, return_qibo=True)
            dm = rl_qc().state()
            per_circ["rl"]["fidelity"].append(float(qibo_fidelity(label, dm)))
            per_circ["rl"]["trace"].append(float(trace_distance(label, dm)))
            per_circ["rl"]["mse"].append(float(mse(label, dm)))

            # RB
            rb_qc = _apply_rb_noise_model(qc, lambda_rb)
            dm = rb_qc().state()  # type: ignore[union-attr]
            per_circ["rb"]["fidelity"].append(float(qibo_fidelity(label, dm)))
            per_circ["rb"]["trace"].append(float(trace_distance(label, dm)))
            per_circ["rb"]["mse"].append(float(mse(label, dm)))

            # No noise
            dm = qc().state()
            per_circ["no_noise"]["fidelity"].append(float(qibo_fidelity(label, dm)))
            per_circ["no_noise"]["trace"].append(float(trace_distance(label, dm)))
            per_circ["no_noise"]["mse"].append(float(mse(label, dm)))

            # MMS
            per_circ["mms"]["fidelity"].append(float(qibo_fidelity(label, mms_dm)))
            per_circ["mms"]["trace"].append(float(trace_distance(label, mms_dm)))
            per_circ["mms"]["mse"].append(float(mse(label, mms_dm)))

        depths_list.append(depth)
        for k in model_keys:
            for metric in ("fidelity", "trace", "mse"):
                vals = np.array(per_circ[k][metric])
                agg[k][metric].append(float(vals.mean()))
                agg[k][f"{metric}_std"].append(float(vals.std()))

    return {"depths": depths_list, **agg}
