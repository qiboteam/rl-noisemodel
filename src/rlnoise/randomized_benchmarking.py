from rlnoise.dataset import Dataset
from rlnoise.circuit_representation import CircuitRepresentation
from rlnoise.noise_model import CustomNoiseModel
import numpy as np
from qibo import Circuit
from qibo import gates
from qibo.backends import NumpyBackend
from qibo.noise import NoiseModel, DepolarizingError
from scipy.optimize import curve_fit
from rlnoise.utils import compute_fidelity, trace_distance

def rb_dataset_generator(config_file):
    """Generate a dataset of circuits for randomized benchmarking."""
    dataset = Dataset(config_file)
    dataset.generate_rb_dataset()
    
def run_rb(rb_dataset, config):
    """Run randomized benchmarking on the circuits in the dataset.
    Return a dictionary with the optimal parameters for the RB model: a, l, b.
    where the model is a * l**depth + b.
    """
    dataset = np.load(rb_dataset, allow_pickle=True)
    circuits = dataset["circuits"]
    print("Preprocessing circuits...")
    circuits = preprocess_circuits(circuits, config)

    return randomized_benchmarking(circuits)

def preprocess_circuits(circuits, config, evaluate=False):
    """Preprocess the circuits for randomized benchmarking.
    Apply noise model and fill the circuit with identity gates where no gate is present."""
    rep = CircuitRepresentation(config)
    noise = CustomNoiseModel(config)

    final_circuits = {}

    for same_len_circuits in circuits:
        for rep_c in same_len_circuits:
            c = rep.rep_to_circuit(rep_c)
            depth = c.depth
            if depth not in final_circuits.keys():
                final_circuits[depth] = []
            nqubits = c.nqubits
            inverse_unitary = gates.Unitary(c.invert().unitary(), *range(nqubits))
            c = fill_identity(c)
            if not evaluate:
                c = noise.apply(c)
                c.add(inverse_unitary)
                for i in range(nqubits):
                    c.add(gates.M(i))
            final_circuits[depth].append(c)
    return final_circuits

def randomized_benchmarking(circuits, nshots=1000):
    """Run randomized benchmarking on the circuits."""
    backend = NumpyBackend()
    nqubits = list(circuits.values())[0][0].nqubits
    probs = { d: [] for d in circuits.keys() }
    init_state = f"{0:0{nqubits}b}"
    
    print('Running randomized benchmarking...')
    for depth, circs in circuits.items():
        print(f'> Looping over circuits of depth: {depth}')
        for c in circs:
            
            result = backend.execute_circuit(c, nshots=nshots)
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
    optimal_params = { 'a': optimal_params[0], 'l': optimal_params[1], 'b': optimal_params[2], "model": "a * l**depth + b" }
    return optimal_params


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

def rb_evaluation(lambda_rb, rb_dataset, config):
    """
    Evaluate the RB model on the circuits in the dataset.
    Return the fidelity and trace distance of the noisy and noiseless circuits.
    The result is a numpy array with the following columns:
    depth, fidelity, fidelity_std, trace_distance, trace_distance_std, 
    fidelity_no_noise, fidelity_no_noise_std, trace_distance_no_noise, trace_distance_no_noise_std
    """
    print("Evaluating RB model...")
    dataset = np.load(rb_dataset, allow_pickle=True)
    circuits = dataset["circuits"]
    labels = dataset["labels"]
    circuits = preprocess_circuits(circuits, config, evaluate=True)
    final_result = []
    depol_noise = NoiseModel()
    depol_noise.add(DepolarizingError(lambda_rb))

    for label_index, circs in enumerate(circuits.values()):
        depth = circs[0].depth
        print(f'> Looping over circuits of depth: {depth}')
        fidelity = []
        trace_dist = []
        fidelity_no_noise = []
        trace_dist_no_noise = []
        for i, c in enumerate(circs):
            dm_no_noise = c().state()
            fidelity_no_noise.append(compute_fidelity(labels[label_index][i], dm_no_noise))
            trace_dist_no_noise.append(trace_distance(labels[label_index][i], dm_no_noise))
            noisy_circuit = depol_noise.apply(c)    
            dm_noise = noisy_circuit().state()
            fidelity.append(compute_fidelity(labels[label_index][i], dm_noise))
            trace_dist.append(trace_distance(labels[label_index][i], dm_noise))
        fidelity_no_noise = np.array(fidelity_no_noise)
        trace_dist_no_noise = np.array(trace_dist_no_noise)
        fidelity = np.array(fidelity)
        trace_dist = np.array(trace_dist)
        result = np.array([(
            depth,
            fidelity.mean(),
            fidelity.std(),
            trace_dist.mean(),
            trace_dist.std(),
            fidelity_no_noise.mean(),
            fidelity_no_noise.std(),
            trace_dist_no_noise.mean(),
            trace_dist_no_noise.std()
            )],
            dtype=[('depth','<f4'),
                    ('fidelity','<f4'),
                    ('fidelity_std','<f4'),
                    ('trace_distance','<f4'),
                    ('trace_distance_std','<f4'),
                    ('fidelity_no_noise','<f4'),
                    ('fidelity_no_noise_std','<f4'),
                    ('trace_distance_no_noise','<f4'),
                    ('trace_distance_no_noise_std','<f4')
                ])
        final_result.append(result)
    
    return np.asarray(final_result)


