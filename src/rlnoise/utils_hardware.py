from collections import Counter

import numpy as np
from qibo import gates, symbols
from qibo.backends import construct_backend, GlobalBackend, NumpyBackend
from qibo.models import Circuit
from qibo.hamiltonians import Hamiltonian
from qiboconnection import API
from qibo.result import MeasurementOutcomes
from qibolab.backends import QibolabBackend
from collections import Counter
from itertools import chain, product
from functools import reduce


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
        self.platform = API(configuration=configuration)
        #self.platform.select_device_id(device_id=device_id)
        self.nqubits = nqubits
        self.qubit_map = qubit_map
        self.device_id = device_id

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
            elif isinstance(gate, gates.I) or len(gate.qubits) == 2:
                new_c.add(gate.__class__(*tuple(qubits), **gate.init_kwargs))
            else:
                matrix = gate.matrix()
                theta, phi, lamb = u3_decomposition(matrix)
                new_c.add([gates.RZ(*tuple(qubits), lamb), gates.RX(*tuple(qubits), np.pi/2), gates.RZ(*tuple(qubits), theta+np.pi), gates.RX(
                    *tuple(qubits), np.pi/2), gates.RZ(*tuple(qubits), phi+np.pi)])  # gates.U3(*tuple(qubits), *u3_decomposition(matrix)))
        return new_c

    def execute_circuit(self, circuits, nshots=1000):
        if isinstance(circuits, list) is False:
            circuits = [circuits]
        for k in range(len(circuits)):
            circuits[k] = self.transpile_circ(circuits[k], self.qubit_map)
        print(nshots)
        results = self.platform.execute_and_return_results(
            circuits, device_id=self.device_id, nshots=nshots, interval=10)[0]
        result_list = []
        for j, result in enumerate(results):
            probs = result['probabilities']
            counts = Counter()
            for key in probs:
                counts[int(key, 2)] = int(probs[key]*nshots)
            result = MeasurementOutcomes(
                circuits[j].measurements, self, nshots=nshots)
            result._frequencies = counts
            result_list.append(result)
        # if len(result_list) == 1:
        #     return result_list[0]
        return result_list
    
class Qibolab_qrc(QibolabBackend):
    def __init__(self, qubit_map=None):
        super().__init__()
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
            elif isinstance(gate, gates.I) or len(gate.qubits) == 2:
                new_c.add(gate.__class__(*tuple(qubits), **gate.init_kwargs))
            else:
                matrix = gate.matrix()
                theta, phi, lamb = u3_decomposition(matrix)
                new_c.add([gates.RZ(*tuple(qubits), lamb), gates.RX(*tuple(qubits), np.pi/2), gates.RZ(*tuple(qubits), theta+np.pi), gates.RX(
                    *tuple(qubits), np.pi/2), gates.RZ(*tuple(qubits), phi+np.pi)])  # gates.U3(*tuple(qubits), *u3_decomposition(matrix)))
        return new_c

    def execute_circuit(self, circuits, nshots=1000):
        if isinstance(circuits, list) is False:
            circuits = [circuits]
        for k in range(len(circuits)):
            circuits[k] = self.transpile_circ(circuits[k], self.qubit_map)
        print(nshots)
        results = self.execute_circuits(circuits, nshots=nshots)
        return results
    

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
    if backend is not None and (backend.name == "QuantumSpain" or backend.name == "qibolab"):
        results = backend.execute_circuit(cal_circs, nshots=nshots)
    else:
        results = [backend.execute_circuit(
            cal_circ, nshots=nshots) for cal_circ in cal_circs]

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
        freqs_mit[format(i, f"0{nqubits}b")] = float(val[0])
    return freqs_mit


class StateTomography:
    def __init__(self, nshots=10000, backend=None):
        self.circuit = None
        self.nqubits = None
        self.backend = backend
        self.tomo_circuits = None
        self.obs = None
        self.exps_vals = None
        self.nshots = nshots
        self.freqs = None
        self.mit_freqs = None

        if backend == None:
            self.backend = GlobalBackend()

    def get_circuits(self, circuit):
        self.circuit = circuit
        self.nqubits = circuit.nqubits
        circuits = []
        self.obs = list(product(['I', 'X', 'Y', 'Z'], repeat=self.nqubits))
        for obs in self.obs[1::]:
            circuit = self.circuit.copy(deep=True)
            for q in range(self.nqubits):
                if obs[q] == 'X':
                    circuit.add([gates.H(0)])
                elif obs[q] == 'Y':
                    circuit.add([gates.S(q).dagger(), gates.H(q)])
            circuit.add(gates.M(*range(self.nqubits)))
            circuits.append(circuit)
        self.tomo_circuits = circuits

        return circuits

    def run_circuits(self):
        dims = np.shape(self.tomo_circuits)
        circs = list(chain.from_iterable(self.tomo_circuits))
        if self.backend is not None and (self.backend.name == "QuantumSpain" or self.backend.name == "qibolab"):
            results = self.backend.execute_circuit(circs, nshots=self.nshots)
        else:
            results = [self.backend.execute_circuit(
                circ, nshots=self.nshots) for circ in circs]

        freqs = [result.frequencies() for result in results]

        freqs = list(np.reshape(freqs, dims))
        self.freqs = freqs

        return freqs

    def _get_cal_mat(self, noise=None):
        self.cal_mat = calibration_matrix(
            self.nqubits, noise_model=noise, nshots=self.nshots, backend=self.backend)

    def redadout_mit(self, freqs, noise=None):
        dims = np.shape(freqs)
        freqs = list(chain.from_iterable(freqs))
        if self.cal_mat is None:
            self._get_cal_mat(noise)
        mit_freqs = []
        for freq in freqs:
            mit_freqs.append(apply_readout_mitigation(freq, self.cal_mat))
        mit_freqs = list(np.reshape(mit_freqs, dims))
        self.mit_freqs = mit_freqs

        return mit_freqs

    def meas_obs(self, noise=None, readout_mit=False):
        exps = []
        for k, circ in enumerate(self.tomo_circuits):
            obs = self.obs[k+1]
            term = np.eye(2**self.nqubits)
            for q in range(self.nqubits):
                if obs[q] != 'I':
                    term = term@symbols.Z(q).full_matrix(self.nqubits)
            obs = Hamiltonian(self.nqubits, term, self.backend)

            if noise is not None and self.backend.name != "QuantumSpain" and self.backend.name != "qibolab":
                circ = noise.apply(circ)

            freqs = self.freqs[k]
            if readout_mit:
                freqs = self.mit_freqs[k]
            # obs.expectation_from_samples(freqs)
            exp = expectation_from_samples(obs, freqs)
            exps.append([self.obs[k], exp])
        self.exps_vals = exps

    def _likelihood(self, mu):
        vals, vecs = np.linalg.eig(mu)
        index = vals.argsort()[::-1]
        vals = vals[index]
        vecs = vecs[:, index]

        lamb = np.zeros(2**self.nqubits, dtype=complex)
        i = 2**self.nqubits
        a = 0

        while vals[i-1] + a/i < 0:
            lamb[i-1] = 0
            a += vals[i-1]
            i -= 1
        for j in range(i):
            lamb[j] = vals[j] + a/i

        rho = 0
        for i in range(2**self.nqubits):
            vec = np.reshape(vecs[:, i], (-1, 1))
            rho += lamb[i]*vec@np.conjugate(np.transpose(vec))

        return rho

    def get_rho(self, likelihood=True):
        I = self.backend.cast(symbols.I(0).full_matrix(1))
        X = self.backend.cast(symbols.X(0).full_matrix(1))
        Y = self.backend.cast(symbols.Y(0).full_matrix(1))
        Z = self.backend.cast(symbols.Z(0).full_matrix(1))
        rho = self.backend.cast(np.eye(2**self.nqubits))
        for k, obs in enumerate(self.obs[1::]):
            obs = list(obs)
            for j, term in enumerate(obs):
                exec('obs[j] =' + term, globals(), locals())
            term = reduce(np.kron, obs)
            term = self.backend.cast(term, term.dtype)
            rho += self.exps_vals[k][1]*term
        rho /= 2**self.nqubits
        if likelihood:
            rho = self._likelihood(rho)

        return rho


def state_tomography(circs, nshots, likelihood, backend):
    st = StateTomography(nshots=nshots, backend=backend)

    tomo_circs = []
    for circ in circs:
        tomo_circs.append(st.get_circuits(circ))
    st.tomo_circuits = tomo_circs

    st._get_cal_mat()
    freqs = st.run_circuits()
    mit_freqs = st.redadout_mit(freqs)

    results = []
    for k, circ in enumerate(circs):
        st.tomo_circuits = np.array(tomo_circs)[k, :]
        st.freqs = np.array(freqs)[k, :]
        st.mit_freqs = np.array(mit_freqs)[k, :]

        st.meas_obs(noise=None, readout_mit=False)
        rho = st.get_rho(likelihood=likelihood)

        st.meas_obs(noise=None, readout_mit=True)
        rho_mit = st.get_rho(likelihood=likelihood)

        circ1 = Circuit(circ.nqubits)
        for gate in circ.queue:
            circ1.add(gate)
        circ1.density_matrix = True
        backend_exact = construct_backend('numpy')
        rho_exact = backend_exact.execute_circuit(circ1).state()

        results.append([circ, rho_exact, rho, rho_mit, st.cal_mat])

    return results
