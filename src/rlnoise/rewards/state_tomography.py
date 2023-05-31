from qibo import gates, symbols
from qibo.backends import GlobalBackend
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.models.error_mitigation import calibration_matrix, apply_readout_mitigation
from itertools import product
import numpy as np


class StateTomography:
    def __init__(self, nshots = 10000, backend = None):
        self.circuit = None
        self.nqubits = None
        self.backend = backend
        self.tomo_circuits = None
        self.obs = None
        self.exps_vals = None
        self.nshots = nshots

        if backend == None:
            self.backend = GlobalBackend()

    def get_circuits(self, circuit):
        self.circuit = circuit
        self.nqubits = circuit.nqubits
        circuits =  []
        self.obs = list(product(['X','Y','Z'], repeat=self.nqubits))
        for obs in self.obs:
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
        sym = 1
        for q in range(self.nqubits):
            sym *= symbols.Z(q)
        obs = SymbolicHamiltonian(sym)
        exps = []
        if readout_mit:
            self.cal_mat = calibration_matrix(self.nqubits,noise_model=noise,backend=self.backend)
        for k, circ in enumerate(self.tomo_circuits):
            if noise is not None:
                circ = noise.apply(circ)
            result = self.backend.execute_circuit(circ, nshots=self.nshots)
            if readout_mit:
                result = apply_readout_mitigation(result, self.cal_mat)
            exp = result.expectation_from_samples(obs)
            exps.append([self.obs[k],exp])
        self.exps_vals = exps

    def get_rho(self):
        Id = symbols.I(0).full_matrix(1)
        X = symbols.X(0).full_matrix(1)
        Y = symbols.Y(0).full_matrix(1)
        Z = symbols.Z(0).full_matrix(1)
        rho = Id
        for k, obs in enumerate(self.obs):
            for sym in obs:
                if len(obs) == 1:
                    if sym == 'Z':
                        obs = Z
                    elif sym == 'Y':
                        obs = Y
                    else:
                        obs = X
            rho += self.exps_vals[k][1]*obs
        return rho/2**self.nqubits
    