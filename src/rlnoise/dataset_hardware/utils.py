from collections import Counter

import numpy as np
from qibo import gates
from qibo.backends import construct_backend
from qibo.config import log
from qibo.models import Circuit
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_experiments.library import MitigatedStateTomography
from qiskit_experiments.library import StateTomography as StateTomography_qiskit


def run_qiskit(circuits, backend, nshots=10000, layout=None):
    circuits_qiskit = []
    for circ in circuits:
        circuits_qiskit.append(QuantumCircuit().from_qasm_str(circ.to_qasm()))
    circuits_qiskit = transpile(circuits_qiskit, initial_layout=layout, optimization_level=0)
    job = backend.run(circuits_qiskit, shots=nshots)
    print(job.status())
    result = job.result()

    counts = []
    for qc in circuits_qiskit:
        counts_qiskit = Counter(result.get_counts(qc))
        counts_qibo = Counter()
        keys = counts_qiskit.keys()
        for k in keys:
            counts_qibo[k[::-1]] = counts_qiskit[k]
        del counts_qiskit
        counts.append(counts_qibo)
    return counts

def run_qibo(circuits, backend, nshots=10000):
    freqs = []
    for circ in circuits:
        result = backend.execute_circuit(circ, nshots=nshots)
        freqs.append(result.frequencies())
    return freqs



def calibration_matrix(nqubits, noise_model=None, nshots: int = 1000, backend=None, backend_qiskit=None, layout=None):
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
    if backend is None:
        from qibo.backends import GlobalBackend

        backend = GlobalBackend()

    matrix = np.zeros((2**nqubits, 2**nqubits))
    
    cal_circs = []
    for i in range(2**nqubits):
        state = format(i, f"0{nqubits}b")

        circuit = Circuit(nqubits, density_matrix=True)
        for q, bit in enumerate(state):
            if bit == "1":
                circuit.add(gates.X(q))
        circuit.add(gates.M(*range(nqubits)))

        if noise_model is not None and backend.name != "qibolab" and backend_qiskit is None:
            circuit = noise_model.apply(circuit)
        cal_circs.append(circuit)

    if backend_qiskit is not None:
        freqs = run_qiskit(cal_circs, backend_qiskit, nshots=nshots, layout=layout)
    else:
        freqs = run_qibo(cal_circs, backend, nshots=nshots)

    for i in range(2**nqubits):
        freq = freqs[i]
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
    keys = list(freqs.keys())
    nqubits = len(keys[0])
    freq = np.zeros(2**nqubits)
    for k, v in freqs.items():
        freq[int(k, 2)] = v

    freq = freq.reshape(-1, 1)
    freqs_mit = Counter()
    for i, val in enumerate(calibration_matrix @ freq):
        freqs_mit[format(i, f"0{nqubits}b")] = float(val)
    return freqs_mit


def rx_rule(gate, platform):
    from qibolab.pulses import PulseSequence

    num = int(gate.parameters[0] / (np.pi/2))
    start = 0
    sequence = PulseSequence()
    for _ in range(num):
        qubit = gate.target_qubits[0]
        RX90_pulse = platform.create_RX90_pulse(
            qubit,
            start=start,
            relative_phase=0,
        )
        sequence.add(RX90_pulse)
        start = RX90_pulse.finish

    return sequence, {}

def x_rule(gate, platform):
    from qibolab.pulses import PulseSequence
    num = 2
    start = 0
    sequence = PulseSequence()
    for _ in range(num):
        qubit = gate.target_qubits[0]
        RX90_pulse = platform.create_RX90_pulse(
            qubit,
            start=start,
            relative_phase=0,
        )
        sequence.add(RX90_pulse)
        start = RX90_pulse.finish

    return sequence, {}


def state_tomography(circs, nshots, likelihood, backend, backend_qiskit, layout):
    from rlnoise.dataset_hardware.state_tomography import StateTomography
    
    st = StateTomography(nshots=nshots,backend=backend, backend_qiskit=backend_qiskit, layout=layout)

    tomo_circs = []
    for circ in circs:
        tomo_circs.append(st.get_circuits(circ))
    st.tomo_circuits = tomo_circs

    st._get_cal_mat()
    freqs = st.run_circuits()
    mit_freqs = st.redadout_mit(freqs)

    results = []
    for k, circ in enumerate(circs):
        st.tomo_circuits = np.array(tomo_circs)[k,:]
        st.freqs =  np.array(freqs)[k,:]
        st.mit_freqs =  np.array(mit_freqs)[k,:]

        st.meas_obs(noise=None,readout_mit=False)
        rho = st.get_rho(likelihood=likelihood)

        st.meas_obs(noise=None,readout_mit=True)
        rho_mit = st.get_rho(likelihood=likelihood)

        circ.density_matrix = True
        backend_exact = construct_backend('numpy')
        rho_exact = backend_exact.execute_circuit(circ).state()
    
        log.info(circ.draw())
        results.append([circ, rho_exact, rho, rho_mit, st.cal_mat])

    return results


def classical_shadows(circ, shadow_size, backend, backend_qiskit):
    from rlnoise.dataset_hardware.classical_shadows import ClassicalShadows
    
    model = ClassicalShadows(circuit=circ,shadow_size=shadow_size)

    model.get_classical_shadow(backend=backend, backend_qiskit=backend_qiskit)
    rho = model.shadow_state_reconstruction()

    circ.density_matrix = True
    backend_exact = construct_backend('numpy')
    rho_exact = backend_exact.execute_circuit(circ).state()

    log.info(circ.draw())
    result = [circ, rho_exact, rho]
    log.info(result)

    return result


def qiskit_state_tomography(circ, nshots, backend):

    circ_qiskit = QuantumCircuit().from_qasm_str(circ.to_qasm()).reverse_bits()


    st = StateTomography_qiskit(circ_qiskit,backend=backend)
    st.set_transpile_options(optimization_level=0)
    results = st.run(backend, shots=nshots)
    rho = results.analysis_results("state").value.to_operator().data


    st = MitigatedStateTomography(circ_qiskit,backend=backend)
    st.set_transpile_options(optimization_level=0)
    results = st.run(backend)
    rho_mit = results.analysis_results("state").value.to_operator().data

    cal_mat = results.analysis_results("Local Readout Mitigator").value.assignment_matrix()

    circ.density_matrix = True
    backend_exact = construct_backend('numpy')
    rho_exact = backend_exact.execute_circuit(circ).state()


    log.info(circ.draw())
    result = [circ, rho_exact, rho, rho_mit, cal_mat]
    log.info(result)

    return result