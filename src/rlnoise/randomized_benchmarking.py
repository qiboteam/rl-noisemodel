from rlnoise.dataset import Dataset
from rlnoise.circuit_representation import CircuitRepresentation
from rlnoise.noise_model import CustomNoiseModel
import numpy as np
import json
from qibo import Circuit
from qibo import gates
from qibo.backends import NumpyBackend, GlobalBackend
from qibo.noise import NoiseModel, DepolarizingError
from scipy.optimize import curve_fit
from rlnoise.utils import compute_fidelity, mse, mms

def rb_dataset_generator(config_file, backend=None):
    """Generate a dataset of circuits for randomized benchmarking."""
    dataset = Dataset(config_file, only_rb=True)
    dataset.generate_rb_dataset(backend)
    
def run_rb(rb_dataset, config, backend=None):
    """Run randomized benchmarking on the circuits in the dataset.
    Return a dictionary with the optimal parameters for the RB model: a, l, b.
    where the model is a * l**depth + b.
    """
    dataset = np.load(rb_dataset, allow_pickle=True)
    circuits = dataset["circuits"]
    circuits = preprocess_circuits(circuits, config, backend=backend)
    if backend is not None and (backend.name == "QuantumSpain" or backend.name == "qibolab"):
        with open(config) as f:
            config = json.load(f)
        nshots = config["chip_conf"]["nshots"]
    else:
        nshots = None

    return randomized_benchmarking(circuits, nshots, backend)

def preprocess_circuits(circuits, config, evaluate=False, backend=None):
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
                if backend is None or (backend.name != "QuantumSpain" and backend.name != "qibolab"):
                    c = noise.apply(c)
                c.add(inverse_unitary)
                for i in range(nqubits):
                    c.add(gates.M(i))
            final_circuits[depth].append(c)
    return final_circuits

def randomized_benchmarking(circuits, nshots=1000, backend=None, verbose=True):
    """Run randomized benchmarking on the circuits."""
    if backend is None:
        backend = GlobalBackend()
    nqubits = list(circuits.values())[0][0].nqubits
    probs = { d: [] for d in circuits.keys() }
    init_state = f"{0:0{nqubits}b}"
    
    print('Running randomized benchmarking...')
    for depth, circs in circuits.items():
        if verbose:
            print(f'> Looping over circuits of depth: {depth}')
        if backend is not None and (backend.name == "QuantumSpain" or backend.name == "qibolab"):
            results = backend.execute_circuit_(circs, nshots=nshots)
        else:
            results = [backend.execute_circuit(circ, nshots=nshots) for circ in circs]

        freq_list = [result.frequencies() for result in results] 
        for freq in freq_list:            
            if init_state not in freq:
                probs[depth].append(0)
            else:
                probs[depth].append(freq[init_state]/nshots)
    avg_probs = [ (d, np.mean(p)) for d,p in probs.items() ]
    std_probs = [ (d, np.std(p)) for d,p in probs.items() ]
    avg_probs = sorted(avg_probs, key=lambda x: x[0])
    std_probs = sorted(std_probs, key=lambda x: x[0])
    model = lambda depth,a,l: a * np.power(l,depth)
    depths, survival_probs = zip(*avg_probs)
    _, err = zip(*std_probs)
    optimal_params, _ = curve_fit(model, depths, survival_probs, maxfev = 2000, p0=[1,0.5])
    optimal_params = { 'a': optimal_params[0], 'l': optimal_params[1], 'b': 0, "model": "a * l**depth + b" }
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

def rb_evaluation(lambda_rb, rb_dataset, config, verbose=False):
    """
    Evaluate the RB model on the circuits in the dataset.
    Return the fidelity and trace distance of the noisy and noiseless circuits.
    The result is a numpy array with the following columns:
    depth, fidelity, fidelity_std, mse, mse_std, 
    fidelity_no_noise, fidelity_no_noise_std, mse_no_noise, mse_no_noise_std,
    fidelity_mms, fidelity_mms_std, mse_mms, mse_mms_std
    """
    print("Evaluating RB model...")
    dataset = np.load(rb_dataset, allow_pickle=True)
    circuits = dataset["circuits"]
    labels = dataset["labels"]
    space_dim = labels[0][0].shape[0]
    circuits = preprocess_circuits(circuits, config, evaluate=True)
    final_result = []
    depol_noise = NoiseModel()
    depol_noise.add(DepolarizingError(lambda_rb))
    mms_ = mms(space_dim)

    for label_index, circs in enumerate(circuits.values()):
        depth = circs[0].depth
        if verbose:
            print(f'> Looping over circuits of depth: {depth}')
        fidelity = []
        mse_ = []
        fidelity_no_noise = []
        mse_no_noise = []
        fidelity_mms = []
        mse_mms = []
        for i, c in enumerate(circs):
            dm_no_noise = c().state()
            # MMS
            fidelity_mms.append(compute_fidelity(labels[label_index][i], mms_))
            mse_mms.append(mse(labels[label_index][i], mms_))
            # No noise
            fidelity_no_noise.append(compute_fidelity(labels[label_index][i], dm_no_noise))
            mse_no_noise.append(mse(labels[label_index][i], dm_no_noise))
            # RB noise
            noisy_circuit = depol_noise.apply(c)    
            dm_noise = noisy_circuit().state()
            fidelity.append(compute_fidelity(labels[label_index][i], dm_noise))
            mse_.append(mse(labels[label_index][i], dm_noise))

        fidelity_no_noise = np.array(fidelity_no_noise)
        mse_no_noise = np.array(mse_no_noise)
        fidelity = np.array(fidelity)
        mse_ = np.array(mse_)
        fidelity_mms = np.array(fidelity_mms)
        mse_mms = np.array(mse_mms)
        result = np.array([(
            depth,
            fidelity.mean(),
            fidelity.std(),
            mse_.mean(),
            mse_.std(),
            fidelity_no_noise.mean(),
            fidelity_no_noise.std(),
            mse_no_noise.mean(),
            mse_no_noise.std(),
            fidelity_mms.mean(),
            fidelity_mms.std(),
            mse_mms.mean(),
            mse_mms.std()
            )],
            dtype=[ ('depth','<f4'),
                    ('fidelity','<f4'),
                    ('fidelity_std','<f4'),
                    ('mse','<f4'),
                    ('mse_std','<f4'),
                    ('fidelity_no_noise','<f4'),
                    ('fidelity_no_noise_std','<f4'),
                    ('mse_no_noise','<f4'),
                    ('mse_no_noise_std','<f4'),
                    ('fidelity_mms','<f4'),
                    ('fidelity_mms_std','<f4'),
                    ('mse_mms','<f4'),
                    ('mse_mms_std','<f4')
                ])
        final_result.append(result)
    
    return np.asarray(final_result)


