from collections import Counter

import numpy as np
from qibo import gates
from qibo.backends import construct_backend
from qibo.models import Circuit
from qiboconnection import API
from qibo.result import MeasurementOutcomes
from collections import Counter

def expectation_from_samples(obs, freq, qubit_map=None):
    obs = obs.matrix
    keys = list(freq.keys())
    if qubit_map is None:
        qubit_map = list(range(int(np.log2(len(obs)))))
    counts = np.array(list(freq.values())) / sum(freq.values())
    expval = 0
    size = len(qubit_map)
    for j, k in enumerate(keys):
        index = 0
        for i in qubit_map:
            index += int(k[qubit_map.index(i)]) * 2 ** (size - 1 - i)
        expval += obs[index, index] * counts[j]
    return np.real(expval)


class QuantumSpain(NumpyBackend):
    def __init__(self, configuration, device_id, nqubits, qubit_map=None):
        super().__init__()
        self.name = "QuantumSpain"
        self.platform = API(configuration = configuration)
        self.platform.select_device_id(device_id=device_id)
        self.nqubits = nqubits
        self.qubit_map = qubit_map

    def transpile_circ(self, circuit, qubit_map=None):
        if qubit_map == None:
            qubit_map = list(range(circuit.nqubits))
        self.qubit_map = qubit_map
        from qibo.transpiler.unitary_decompositions import u3_decomposition
        new_c = Circuit(self.nqubits, density_matrix=True)
        for gate in circuit.queue:
            qubits = [self.qubit_map[j] for j in gate.qubits]
            if isinstance(gate, gates.M):
                new_gate = gates.M(*tuple(qubits), **gate.init_kwargs)
                new_gate.result = gate.result
                new_c.add(new_gate)
            elif isinstance(gate, gates.I):
                new_c.add(gate.__class__(*tuple(qubits), **gate.init_kwargs))
            else:
                matrix = gate.matrix()
                theta, phi, lamb = u3_decomposition(matrix)
                new_c.add([gates.RZ(*tuple(qubits),lamb),gates.RX(*tuple(qubits),np.pi/2),gates.RZ(*tuple(qubits),theta+np.pi),gates.RX(*tuple(qubits),np.pi/2),gates.RZ(*tuple(qubits),phi+np.pi)])#gates.U3(*tuple(qubits), *u3_decomposition(matrix)))
        return new_c
    
    def execute_circuit(self, circuits, nshots=1000):
        if isinstance(circuits, list) is False:
            circuits = [circuits]
        for k in range(len(circuits)):
            circuits[k] = self.transpile_circ(circuits[k], self.qubit_map)
        results = self.platform.execute_and_return_results(circuits, nshots=nshots, interval=10)[0]
        result_list = []
        for j, result in enumerate(results):
            probs = result['probabilities']
            counts = Counter()
            for key in probs:
                counts[int(key,2)] = int(probs[key]*nshots)
            result = MeasurementOutcomes(circuits[j].measurements, self, nshots=nshots)
            result._frequencies = counts
            result_list.append(result)
        # if len(result_list) == 1:
        #     return result_list[0]
        return result_list

def calibration_matrix(nqubits, noise_model=None, nshots: int = 1000, backend=None):
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

        if noise_model is not None:
            circuit = noise_model.apply(circuit)
        cal_circs.append(circuit)

    if backend is not None and backend.name == "QuantumSpain":
        results = backend.execute_circuit(cal_circs, nshots=nshots)
    else:
        results = [backend.execute_circuit(cal_circ, nshots=nshots) for cal_circ in cal_circs]

    freqs = [result.frequencies() for result in results]

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


def state_tomography(circs, nshots, likelihood, backend):
    from hardware.state_tomography import StateTomography
    
    st = StateTomography(nshots=nshots,backend=backend)

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

        circ1 = Circuit(circ.nqubits)
        for gate in circ.queue:
            circ1.add(gate)
        circ1.density_matrix = True
        backend_exact = construct_backend('numpy')
        rho_exact = backend_exact.execute_circuit(circ1).state()
    
        results.append([circ, rho_exact, rho, rho_mit, st.cal_mat])

    return results