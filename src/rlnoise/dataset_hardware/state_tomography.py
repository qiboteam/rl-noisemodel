from qibo import gates, symbols
from qibo.backends import GlobalBackend
from qibo.hamiltonians import Hamiltonian
#from qibo.models.error_mitigation import calibration_matrix, apply_readout_mitigation
from itertools import product
from functools import reduce
import numpy as np
from rlnoise.dataset_hardware.utils import run_qiskit, calibration_matrix, apply_readout_mitigation

class StateTomography:
    def __init__(self, nshots = 10000, backend = None, backend_qiskit = None, layout=None):
        self.circuit = None
        self.nqubits = None
        self.backend = backend
        self.tomo_circuits = None
        self.obs = None
        self.exps_vals = None
        self.nshots = nshots
        self.backend_qiskit = backend_qiskit
        self.layout = layout

        if backend == None:
            self.backend = GlobalBackend()

    def get_circuits(self, circuit):
        self.circuit = circuit
        self.nqubits = circuit.nqubits
        circuits =  []
        self.obs = list(product(['I','X','Y','Z'], repeat=self.nqubits))
        for obs in self.obs[1::]:
            circuit = self.circuit.copy(deep=True)
            for q in range(self.nqubits):
                if obs[q] == 'X':
                    circuit.add([gates.RZ(q,np.pi/2),gates.RX(q,np.pi/2),gates.RZ(q,np.pi/2)]) #H
                elif obs[q] == 'Y':
                    circuit.add([gates.RX(q,np.pi/2),gates.RZ(q,np.pi/2)]) #S^tH
            circuit.add(gates.M(*range(self.nqubits)))
            circuits.append(circuit)
        self.tomo_circuits = circuits
        return circuits
    
    def meas_obs(self, noise = None, readout_mit = False):
        exps = []
        if readout_mit:
            self.cal_mat = calibration_matrix(self.nqubits,noise_model=noise,backend=self.backend,backend_qiskit=self.backend_qiskit,layout=self.layout)
        if self.backend_qiskit is not None:
            freqs_list = run_qiskit(self.tomo_circuits, self.backend_qiskit, self.nshots, self.layout)

        for k, circ in enumerate(self.tomo_circuits):
            obs = self.obs[k+1]
            term = np.eye(2**self.nqubits)
            for q in range(self.nqubits):
                if obs[q] != 'I':
                    term = term@symbols.Z(q).full_matrix(self.nqubits)            
            obs = Hamiltonian(self.nqubits,term,self.backend)

            if noise is not None and self.backend_qiskit is None:
                circ = noise.apply(circ)

            if self.backend_qiskit is not None:
                freqs = freqs_list[k]
            else:
                result = self.backend.execute_circuit(circ, nshots=self.nshots)
                freqs = result.frequencies()
            if readout_mit:
                freqs = apply_readout_mitigation(freqs, self.cal_mat)
            exp = obs.expectation_from_samples(freqs)
            exps.append([self.obs[k],exp])
        self.exps_vals = exps

    def _likelihood(self, mu):
        vals, vecs = np.linalg.eig(mu)
        index = vals.argsort()[::-1]
        vals = vals[index]
        vecs = vecs[index]

        lamb = np.zeros(2**self.nqubits,dtype=complex)
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
            vec = np.reshape(vecs[i,:],(-1,1))
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
            term = reduce(np.kron,obs)
            term = self.backend.cast(term,term.dtype)
            rho += self.exps_vals[k][1]*term
        rho /=2**self.nqubits
        if likelihood:
            rho = self._likelihood(rho)
        return rho
    