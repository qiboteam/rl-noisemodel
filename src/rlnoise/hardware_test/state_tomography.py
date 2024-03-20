from qibo import gates, symbols
from qibo.backends import GlobalBackend
from qibo.hamiltonians import Hamiltonian
from itertools import product
from functools import reduce
import numpy as np
from rlnoise.hardware_test.utils import run_qiskit, run_qibo, run_quantum_spain, calibration_matrix, apply_readout_mitigation, expectation_from_samples
from itertools import chain

class StateTomography:
    def __init__(self, nshots = 10000, backend = None, backend_qiskit = None, backend_qs = None, layout=None):
        self.circuit = None
        self.nqubits = None
        self.backend = backend
        self.tomo_circuits = None
        self.obs = None
        self.exps_vals = None
        self.nshots = nshots
        self.backend_qiskit = backend_qiskit
        self.backend_qs = backend_qs
        self.layout = layout
        self.freqs = None
        self.mit_freqs = None

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
                    circuit.add([gates.H(0)])#[gates.RZ(q,np.pi/2),gates.SX(q),gates.RZ(q,np.pi/2)]) #H
                elif obs[q] == 'Y':
                    # basis_y = 1/np.sqrt(2)*np.ones((2,2),complex)
                    # basis_y[0,1] = -basis_y[0,1]*1j
                    # basis_y[1,1] = basis_y[1,1]*1j
                    circuit.add([gates.S(q).dagger(),gates.H(q)])#[gates.Unitary(basis_y,q)])#[gates.SX(q),gates.RZ(q,np.pi/2)]) #S^tH
            circuit.add(gates.M(*range(self.nqubits)))
            circuits.append(circuit)
        self.tomo_circuits = circuits

        return circuits
    
    def run_circuits(self):
        dims = np.shape(self.tomo_circuits)
        circs = list(chain.from_iterable(self.tomo_circuits))

        if self.backend_qiskit is not None:
            freqs_list = run_qiskit(circs, self.backend_qiskit, self.nshots, self.layout)
        if self.backend_qs is not None:
<<<<<<< HEAD
            freqs_list = run_quantum_spain(circs, self.backend_qs, self.nshots)
=======
            freqs_list = run_quantum_spain(circs, self.backend_qs, self.nshots, self.layout)
>>>>>>> main
        else:
            freqs_list = run_qibo(circs, self.backend, self.nshots)

        freqs_list = list(np.reshape(freqs_list,dims))
        self.freqs = freqs_list

        return freqs_list
    
    def _get_cal_mat(self, noise=None):
        self.cal_mat = calibration_matrix(self.nqubits,noise_model=noise,backend=self.backend,backend_qiskit=self.backend_qiskit,backend_qs=self.backend_qs,layout=self.layout)
    
    def redadout_mit(self, freqs, noise = None):
        dims = np.shape(freqs)
        freqs = list(chain.from_iterable(freqs))
        if self.cal_mat is None:
            self._get_cal_mat(noise)
        mit_freqs = []
        for freq in freqs:
            mit_freqs.append(apply_readout_mitigation(freq, self.cal_mat))
        mit_freqs = list(np.reshape(mit_freqs,dims))
        self.mit_freqs = mit_freqs

        return mit_freqs
    
    def meas_obs(self, noise = None, readout_mit = False):
        exps = []   
        for k, circ in enumerate(self.tomo_circuits):
            obs = self.obs[k+1]
            term = np.eye(2**self.nqubits)
            for q in range(self.nqubits):
                if obs[q] != 'I':
                    term = term@symbols.Z(q).full_matrix(self.nqubits)            
            obs = Hamiltonian(self.nqubits,term,self.backend)

            if noise is not None and self.backend_qiskit is None:
                circ = noise.apply(circ)

            freqs = self.freqs[k]
            if readout_mit:
                freqs = self.mit_freqs[k]
            exp =  expectation_from_samples(obs, freqs)#obs.expectation_from_samples(freqs)
            exps.append([self.obs[k],exp])
        self.exps_vals = exps

    def _likelihood(self, mu):
        vals, vecs = np.linalg.eig(mu)
        index = vals.argsort()[::-1]
        vals = vals[index]
        vecs = vecs[:,index]

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
            vec = np.reshape(vecs[:,i],(-1,1))
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