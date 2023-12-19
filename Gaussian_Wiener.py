import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from rlnoise.custom_noise import CustomNoiseModel
from qibo.models.encodings import unary_encoder_random_gaussian
from qibo.models.qft import QFT
from rlnoise.utils import unroll_circuit
from qibo.result import QuantumState, MeasurementOutcomes

from qibo import Circuit, gates
from qibo.models.grover import Grover

def check_single_cz(circuit: Circuit):
    for moments in circuit.queue.moments:
        num_cz = len([1 for gate in moments if type(gate) == gates.CZ])/2
        if num_cz > 1:
            raise ValueError(f"More than one CZ at moment {moments}")

def plot_gaussian_hist(n_qubits: int = 4, n_samples: int = 1000):
    results_list = []
    for _ in range(n_samples):
        gaussian_circ = unary_encoder_random_gaussian(n_qubits)
        gaussian_circ.add(gates.M(*range(gaussian_circ.nqubits)))
        results_list += [value for value in gaussian_circ().state().real if value !=0]
    results = np.array(results_list).flatten()
    plt.hist(results, bins = 50, density=True)
    plt.show()

    
    
def main(**args):
    nqubits = args["nqubits"]
    gaussian_circuit = unary_encoder_random_gaussian(nqubits)
    qft_circuit = unroll_circuit(QFT(nqubits, with_swaps=False))
    #ADD NOISE TO QFT &/or gaussian_circuit (to add also on gaussian we need to unroll it!)
    wiener_circuit = gaussian_circuit + qft_circuit
    check_single_cz(wiener_circuit)
    # plot gaussian
    # plot_gaussian_hist(nqubits)

    
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nqubits", default=8, type=int, help="Number of qubits.")
    args = vars(parser.parse_args())
    main(**args)