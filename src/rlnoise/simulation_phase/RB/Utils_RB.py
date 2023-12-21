from tqdm import tqdm
from scipy.optimize import curve_fit
from qibo import gates, Circuit
import numpy as np
from copy import deepcopy
from qibo.models.error_mitigation import apply_readout_mitigation
from qibo.models.error_mitigation import calibration_matrix as cm


def randomized_benchmarking(circuits, backend=None, nshots=1000, noise_model=None):
    nqubits = circuits[0].nqubits
    calibration_matrix = cm(nqubits, backend=backend, noise_model=noise_model, nshots=nshots)
    circuits = [ deepcopy(c) for c in circuits ] 
    if backend is None:
        from qibo.backends import GlobalBackend
        backend = GlobalBackend()
    
    circ = { c.depth: [] for c in circuits }
    for c in tqdm(circuits):
        depth = c.depth
        inverse_unitary = gates.Unitary(c.invert().unitary(), *range(nqubits))
        c = fill_identity(c)
        if noise_model is not None:
            c = noise_model.apply(c)
        c.add(inverse_unitary)
        circ[depth].append(c)        
        
    probs = { d: [] for d in circ.keys() }
    init_state = f"{0:0{nqubits}b}"
    for depth, circs in tqdm(circ.items()):
        print(f'> Looping over circuits of depth {depth}')
        for c in circs:
            for i in range(nqubits):
                c.add(gates.M(i))
            result = apply_readout_mitigation(backend.execute_circuit(c, nshots=nshots), calibration_matrix)
            freq = result.frequencies()
            
            if init_state not in freq:
                probs[depth].append(0)
            else:
                probs[depth].append(freq[init_state]/nshots)
    avg_probs = [ (d, np.mean(p)) for d,p in probs.items() ]
    std_probs = [ (d, np.std(p)) for d,p in probs.items() ]
    avg_probs = sorted(avg_probs, key=lambda x: x[0])
    std_probs = sorted(std_probs, key=lambda x: x[0])
    model = lambda depth,a,l,b: a * np.power(l,depth) + b
    depths, survival_probs = zip(*avg_probs)
    _, err = zip(*std_probs)
    optimal_params, _ = curve_fit(model, depths, survival_probs, maxfev = 2000, p0=[1,0.5,0])
    model = lambda depth: optimal_params[0] * np.power(optimal_params[1],depth) + optimal_params[2]
    return depths, survival_probs, err, optimal_params, model


def fill_identity(circuit: Circuit):
    """Fill the circuit with identity gates where no gate is present to apply RB noisemodel.
    Works with circuits with no more than 3 qubits."""
    new_circuit = circuit.__class__(**circuit.init_kwargs)
    for moment in circuit.queue.moments:
        f=0
        for qubit, gate in enumerate(moment):
            if gate is not None:
                if gate.__class__ is gates.CZ and f==0:
                    new_circuit.add(gate)
                    f=1
                elif not gate.__class__ is gates.CZ:
                    new_circuit.add(gate)
            else:
                new_circuit.add(gates.I(qubit))
    return new_circuit