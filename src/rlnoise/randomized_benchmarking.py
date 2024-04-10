from rlnoise.dataset import Dataset
from rlnoise.circuit_representation import CircuitRepresentation
from rlnoise.noise_model import CustomNoiseModel
import numpy as np
from qibo import Circuit
from qibo.noise import NoiseModel, DepolarizingError

def rb_dataset_generator(config_file):
    dataset = Dataset(config_file)
    dataset.generate_rb_dataset()
    
def run_rb(rb_dataset, config):
    rep = CircuitRepresentation(config)
    noise = CustomNoiseModel(config)
    dataset = np.load(rb_dataset, allow_pickle=True)
    circuits = dataset["circuits"]
    labels = dataset["labels"]



    noise_model = None if args.backend == 'qibolab' else CustomNoiseModel(args.config)

    depths, survival_probs, err, optimal_params, model = randomized_benchmarking(circuits, noise_model=noise_model)

    with open('RB.json', 'w') as f:
        json.dump({"depths": depths, "survival probs": survival_probs, "errors": err, "optimal params": optimal_params.tolist()}, f, indent=2)

    import matplotlib.pyplot as plt
    plt.errorbar(depths, survival_probs, yerr=err, fmt="o", elinewidth=1, capsize=3, c='orange')
    plt.plot(depths, model(depths), c='orange')

    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color='orange', label=f"True Noise, Decay: {optimal_params[1]:.2f}")]
    from qibo.backends import NumpyBackend

    # Build a Depolarizing toy model
    depolarizing_toy_model = NoiseModel()
    depolarizing_toy_model.add(DepolarizingError(1 - optimal_params[1]))
    _, survival_probs, err, optimal_params, model = randomized_benchmarking(circuits, noise_model=depolarizing_toy_model, backend=NumpyBackend())

    plt.errorbar(depths, survival_probs, yerr=err, fmt="o", elinewidth=1, capsize=3, c='blue')
    plt.plot(depths, model(depths), c='blue')
    patches.append(mpatches.Patch(color='blue', label=f"Depolarizing toy model, Decay: {optimal_params[1]:.2f}"))

    if args.agent is not None:
        agent = PPO.load(args.agent)
        agent_noise_model = RL_NoiseModel(agent, rep)
        _, survival_probs, err, optimal_params, model = randomized_benchmarking(circuits, noise_model=agent_noise_model, backend=NumpyBackend())
        plt.errorbar(depths, survival_probs, yerr=err, fmt="o", elinewidth=1, capsize=3, c='green')
        plt.plot(depths, model(depths), c='green')
        patches.append(mpatches.Patch(color='green', label=f"RL Agent, Decay: {optimal_params[1]:.2f}"))

    plt.legend(handles=patches)

    plt.ylabel('Survival Probability')
    plt.xlabel('Depth')
    plt.savefig('RB.pdf', format='pdf', dpi=300)
    plt.show()


def randomized_benchmarking(circuits, backend=None, nshots=1000, noise_model=None):
    nqubits = circuits[0].nqubits
    calibration_matrix = cm(nqubits, backend=backend, noise_model=noise_model, nshots=nshots)
    circuits = [ c.copy(True) for c in circuits ] 
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

