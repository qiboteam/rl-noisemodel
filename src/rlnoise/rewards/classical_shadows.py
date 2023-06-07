import numpy as np
from qibo import gates, symbols
from qibo.backends import GlobalBackend
from rlnoise.rewards.utils import run_qiskit

class ClassicalShadows:
    def __init__(self, circuit, shadow_size):
        self.circuit = circuit
        self.shadow_size = shadow_size
        self.shadows = None
    def get_classical_shadow(self, backend=None, backend_qiskit=None, layout=None):

        if backend == None:
            self.backend = GlobalBackend()

        num_qubits = self.circuit.nqubits
        unitary_ensemble = [symbols.X, symbols.Y, symbols.Z]
        unitary_ids = np.random.randint(0, 3, size=(self.shadow_size, num_qubits))
        outcomes = np.zeros((self.shadow_size, num_qubits))

        circuits = []
        for k in range(self.shadow_size):
            circuit = self.circuit.copy(True)
            for i in range(num_qubits):
                mat = unitary_ensemble[int(unitary_ids[k, i])](i)
                if mat.name[0] == 'X':
                    circuit.add(gates.H(i))
                elif mat.name[0] == 'Y':
                    circuit.add([gates.S(i).dagger(),gates.H(i)])
            circuit.add(gates.M(*range(num_qubits)))
            circuits.append(circuit)
            if backend_qiskit is None:
                sample = backend.execute_circuit(circuit,nshots=1).samples()[0]
                for i in range(len(sample)):
                    if sample[i] == 0:
                        sample[i] = 1
                    elif sample[i] == 1:
                        sample[i] = -1
                outcomes[k, :] = sample
        if backend_qiskit is not None:
            samples = run_qiskit(circuits, backend_qiskit, 1, layout)
        for k in range(self.shadow_size):
            sample = list(list(samples[k].keys())[0])
            for i in range(len(sample)):
                sample[i] = int(sample[i])
                if sample[i] == 0:
                    sample[i] = 1
                elif sample[i] == 1:
                    sample[i] = -1
            outcomes[k, :] = sample
        self.shadows = (outcomes, unitary_ids)
    def get_snapshot_state(self, b_list, obs_list):
        num_qubits = len(b_list)
        zero_state = np.array([[1, 0], [0, 0]])
        one_state = np.array([[0, 0], [0, 1]])

        unitaries = [gates.H(0).matrix, gates.H(0).matrix@gates.S(0).matrix, gates.I(0).matrix]

        rho_snapshot = [1]
        for i in range(num_qubits):
            state = zero_state if b_list[i] == 1 else one_state
            U = unitaries[int(obs_list[i])]

            local_rho = 3 * (U.conj().T @ state @ U) - unitaries[2]
            rho_snapshot = np.kron(rho_snapshot, local_rho)
        
        return rho_snapshot
    def shadow_state_reconstruction(self):
        num_snapshots, num_qubits = self.shadows[0].shape
        b_lists, obs_lists = self.shadows

        shadow_rho = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex)
        for i in range(num_snapshots):
            shadow_rho += self.get_snapshot_state(b_lists[i], obs_lists[i])
        
        return shadow_rho / num_snapshots
    def estimate_shadow_obervable(self, observable, k=10):
        shadow_size, _ = self.shadows[0].shape
        map_name_to_int = {"X": 0, "Y": 1, "Z": 2}
        target_obs, target_locs = np.array([map_name_to_int[o.name[0]] for o in observable.args[:-1]]), np.array([o.target_qubit for o in observable.args[:-1]])
        b_lists, obs_lists = self.shadows
        means = []
        for i in range(0, shadow_size, shadow_size // k):
            b_lists_k, obs_lists_k = (
                b_lists[i: i + shadow_size // k],
                obs_lists[i: i + shadow_size // k],
            )
            indices = np.all(obs_lists_k[:, target_locs] == target_obs, axis=1)
            if sum(indices) > 0:
                product = np.prod(b_lists_k[indices][:, target_locs], axis=1)
                means.append(np.sum(product) / sum(indices))
            else:
                means.append(0)
        return np.median(means)
    
def shadow_bound(error, observables, failure_rate=0.01):
    M = len(observables)
    K = 2 * np.log(2 * M / failure_rate)
    shadow_norm = (
        lambda op: np.linalg.norm(
            op - np.trace(op) / 2 ** int(np.log2(op.shape[0])), ord=np.inf
        )
        ** 2
    )
    N = 34 * max(shadow_norm(o) for o in observables) / error ** 2
    return int(np.ceil(N * K)), int(K)
