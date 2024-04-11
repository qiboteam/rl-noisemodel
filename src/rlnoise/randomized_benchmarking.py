from rlnoise.dataset import Dataset
from rlnoise.circuit_representation import CircuitRepresentation
from rlnoise.noise_model import CustomNoiseModel
import numpy as np
from qibo import Circuit
from qibo import gates
from qibo.backends import NumpyBackend
from qibo.noise import NoiseModel, DepolarizingError
from scipy.optimize import curve_fit

def rb_dataset_generator(config_file):
    dataset = Dataset(config_file)
    dataset.generate_rb_dataset()
    
def run_rb(rb_dataset, config):
    dataset = np.load(rb_dataset, allow_pickle=True)
    circuits = dataset["circuits"]
    circuits = preprocess_circuits(circuits, config)

    return randomized_benchmarking(circuits)

def preprocess_circuits(circuits, config):
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
            c = noise.apply(c)
            c.add(inverse_unitary)
            for i in range(nqubits):
                c.add(gates.M(i))
            final_circuits[depth].append(c)
    return final_circuits

def randomized_benchmarking(circuits, nshots=1000):
    
    backend = NumpyBackend()
    nqubits = list(circuits.values())[0][0].nqubits
    probs = { d: [] for d in circuits.keys() }
    init_state = f"{0:0{nqubits}b}"
    
    for depth, circs in circuits.items():
        print(f'> Looping over circuits of depth {depth}')
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

def RB_evaluation(lambda_RB,circ_representation,target_label):
    dataset_size = len(target_label)
    trace_distance_rb_list = []
    bures_distance_rb_list = []
    fidelity_rb_list = []
    trace_distance_no_noise_list = []
    bures_distance_no_noise_list = []
    fidelity_no_noise_list = []
    rb_noise_model=CustomNoiseModel("src/rlnoise/config.json")
    RB_label = np.array([rb_noise_model.apply(CircuitRepresentation().rep_to_circuit(circ_representation[i]))().state() 
                         for i in range(dataset_size)])
    label_no_noise_added = np.array([CircuitRepresentation().rep_to_circuit(circ_representation[i])().state() 
                         for i in range(dataset_size)])
    for idx,label in enumerate(RB_label):
        fidelity_rb_list.append(compute_fidelity(label,target_label[idx]))
        trace_distance_rb_list.append(trace_distance(label,target_label[idx]))
        bures_distance_rb_list.append(bures_distance(label,target_label[idx]))
        fidelity_no_noise_list.append(compute_fidelity(label_no_noise_added[idx],target_label[idx]))
        trace_distance_no_noise_list.append(trace_distance(label_no_noise_added[idx],target_label[idx]))
        bures_distance_no_noise_list.append(bures_distance(label_no_noise_added[idx],target_label[idx]))
    fidelity = np.array(fidelity_rb_list)
    trace_dist = np.array(trace_distance_rb_list)
    bures_dist = np.array(bures_distance_rb_list)
    no_noise_fidelity = np.array(fidelity_no_noise_list)
    no_noise_trace_dist = np.array(trace_distance_no_noise_list)
    no_noise_bures_dist = np.array(bures_distance_no_noise_list)
    results = np.array([(
                       fidelity.mean(),fidelity.std(),
                       trace_dist.mean(),trace_dist.std(),
                       bures_dist.mean(),bures_dist.std(),
                       no_noise_fidelity.mean(),no_noise_fidelity.std(),
                       no_noise_trace_dist.mean(),no_noise_trace_dist.std(),
                       no_noise_bures_dist.mean(),no_noise_bures_dist.std()  )],
                       dtype=[
                              ('fidelity','<f4'),('fidelity_std','<f4'),
                              ('trace_distance','<f4'),('trace_distance_std','<f4'),
                              ('bures_distance','<f4'),('bures_distance_std','<f4'),
                              ('fidelity_no_noise','<f4'),('fidelity_no_noise_std','<f4'),
                              ('trace_distance_no_noise','<f4'),('trace_distance_no_noise_std','<f4'),
                              ('bures_distance_no_noise','<f4'),('bures_distance_no_noise_std','<f4')  ])
    
    return results


