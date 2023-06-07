from qiskit import QuantumCircuit, execute
from collections import Counter
from qibo import gates
from qibo.models import Circuit
import numpy as np


def run_qiskit(circuits, backend, nshots=10000, layout=None):
    for k, circ in enumerate(circuits):
        circuits[k] = QuantumCircuit().from_qasm_str(circ.to_qasm())
        
    job = execute(circuits,shots=nshots,backend=backend,initial_layout=layout,optimization_level=0)
    result = job.result()
    counts = []
    for qc in circuits:
        counts_qiskit = Counter(result.get_counts(qc))
        counts_qibo = Counter()
        keys = counts_qiskit.keys()
        for k in keys:
            counts_qibo[k[::-1]] = counts_qiskit[k]
        del counts_qiskit
        counts.append(counts_qibo)
    return counts


def calibration_matrix(nqubits, noise_model=None, nshots: int = 1000, backend=None, qiskit=False, layout=None):
    """Computes the calibration matrix for readout mitigation.

    Args:
        nqubits (int): Total number of qubits.
        noise_model (:class:`qibo.noise.NoiseModel`, optional): noise model used for simulating
            noisy computation. This matrix can be used to mitigate the effect of
            `qibo.noise.ReadoutError`.
        nshots (int, optional): number of shots.
        backend (:class:`qibo.backends.abstract.Backend`, optional): calculation engine.

    Returns:
        numpy.ndarray : The computed (`nqubits`, `nqubits`) calibration matrix for
            readout mitigation.
    """
    if qiskit is False and backend is None:
        from qibo.backends import GlobalBackend

        backend = GlobalBackend()

    matrix = np.zeros((2**nqubits, 2**nqubits))

    for i in range(2**nqubits):
        state = format(i, f"0{nqubits}b")

        circuit = Circuit(nqubits, density_matrix=True)
        for q, bit in enumerate(state):
            if bit == "1":
                circuit.add(gates.X(q))
        circuit.add(gates.M(*range(nqubits)))

        if noise_model is not None and backend.name != "qibolab" and qiskit is False:
            circuit = noise_model.apply(circuit)
        if qiskit:
            freq = run_qiskit(circuit, backend, nshots=nshots, layout=layout)[0]
        else:
            freq = backend.execute_circuit(circuit, nshots=nshots).frequencies()

        column = np.zeros(2**nqubits)
        for key in freq.keys():
            f = freq[key] / nshots
            column[int(key, 2)] = f
        matrix[:, i] = column

    return np.linalg.inv(matrix)


def apply_readout_mitigation(freqs, calibration_matrix):
    """Updates the frequencies of the input state with the mitigated ones obtained with
    ``calibration_matrix * state.frequencies()``.

    Args:
        state (:class:`qibo.states.CircuitResult`): input state to be updated.
        calibration_matrix (numpy.ndarray): calibration matrix for readout mitigation.

    Returns:
        :class:`qibo.states.CircuitResult`: the input state with the updated frequencies.
    """
    keys = freqs.keys()
    nqubits = len(keys[0])
    freq = np.zeros(2**nqubits)
    for k, v in freqs.items():
        freq[int(k, 2)] = v

    freq = freq.reshape(-1, 1)
    freqs_mit = Counter()
    for i, val in enumerate(calibration_matrix @ freq):
        freqs_mit["{0:b}".format(i)] = float(val)

    return freqs_mit
