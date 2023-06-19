import os
import json
import copy
from rlnoise.custom_noise import CustomNoiseModel
from rlnoise.dataset import CircuitRepresentation
from rlnoise.rewards.rewards import DensityMatrixReward
from qibo.quantum_info import trace_distance
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from rlnoise.gym_env import QuantumCircuit
from qibo import gates
from qibo.quantum_info import trace_distance
from qibo.models.circuit import Circuit
from scipy.linalg import sqrtm

config_path=str(Path().parent.absolute())+'/src/rlnoise/config.json'
with open(config_path) as f:
    config = json.load(f)

DEBUG=False
np.set_printoptions(precision=3, suppress=True)

def models_folder():
    folder = os.path.join(os.getcwd(), "models")
    return folder

def dataset_folder():
    folder = os.path.join(os.getcwd(), "dataset")
    return folder

def figures_folder():
    folder = os.path.join(os.getcwd(), "figures")
    return folder

def moments_matching(m1, m2, v1, v2, alpha=100.):
    '''Moments matching with one hyperparameter, not a reward
    Args:
        m1, m2 (float): mean values
        v1, v2 (float): variance values
        alpha (float): hyperparameter
    '''
    return (m1-m2)**2+alpha*(v1-v2)**2

def neg_moments_matching(m1, m2, v1, v2, alpha=100.):
    '''Moments matching with one hyperparameter (reward)
    Args:
        m1, m2 (float): mean values
        v1, v2 (float): variance values
        alpha (float): hyperparameter
    '''
    return -moments_matching(m1, m2, v1, v2, alpha=alpha)

def truncated_moments_matching(m1, m2, v1, v2, alpha=20., truncate=0.001, normalize=True):
    '''Positive moments matching truncated
    Args:
        m1, m2 (float): mean values
        v1, v2 (float): variance values
        alpha (float): hyperparameter
        truncate (float): maximum value of moments matching
        normalize (bool): normalize reward in [0;1]
    '''
    result=moments_matching(m1, m2, v1, v2, alpha=alpha)
    if result < truncate:
        if normalize:
            return (truncate-result)/truncate
        else:
            return truncate-result
    else:
        return 0.

def kld(m1, m2, v1, v2):
    '''Symmetric KL divergence of two Gaussians, not a reward
    Args:
        m1, m2 (float): mean values
        v1, v2 (float): variance values
        '''
    return 0.5*((m1-m2)**2+(v1+v2))*(1/v1+1/v2)-2

def neg_kld_reward(m1, m2, v1, v2):
    '''Negative Symmetric KL divergence to be used as reward
    Args:
        m1, m2 (float): mean values
        v1, v2 (float): variance values
    '''
    return -kld(m1, m2, v1, v2)

def truncated_kld_reward(m1, m2, v1, v2, truncate=10):
    '''Positive S-KLD truncated
    Args:
        m1, m2 (float): mean values
        v1, v2 (float): variance values
        truncate (float): maximum value of S_KLD
    '''
    result=kld(m1, m2, v1, v2)
    if result < truncate:
        return truncate-result
    else:
        return 0.

def plot_results(train_history, val_history, n_steps=20, filename="train_info.png"): 
    tot_steps=len(train_history)
    train_reward_history=[]
    for index in range(0, tot_steps, n_steps):
        avg_reward=0
        for j in range(index, index+n_steps):
            avg_reward+=train_history[j]["reward"]
        train_reward_history.append(avg_reward/n_steps)

    plt.plot(train_reward_history, c='red', label='train_reward')
    plt.plot(val_history, c='blue', label='validation_reward')
    plt.legend()
    plt.show()
    plt.savefig(figures_folder()+ '/' +filename)

def compute_fidelity(density_matrix0, density_matrix1):
    """Compute the fidelity for two density matrices (pure or mixed states).

    .. math::
            F( \rho , \sigma ) = -\text{Tr}( \sqrt{\sqrt{\rho} \sigma \sqrt{\rho}})^2
    """
    sqrt_mat1_mat2 = sqrtm(density_matrix0 @ density_matrix1)
    trace = np.real(np.trace(sqrt_mat1_mat2)**2)
    if trace > 1:
        trace=1 #TODO: problem the trace can be sligtlhy > 1! This problem appeared only on the hardware test, so probably the dm matrices are not perfect
    return trace

def bures_distance(density_matrix0, density_matrix1):
    """ Compute the Bures distance between density matrices
    .. math::
        B( \rho , \sigma ) = -\sqrt{2*(1-sqrt(F(\sigma,\rho)))} where F is the fidelity
    """
    return np.sqrt(2*(1-np.sqrt(compute_fidelity(density_matrix0, density_matrix1))))

def model_evaluation(evaluation_circ,evaluation_labels,model,reward=DensityMatrixReward(),representation=CircuitRepresentation()):
    '''
    Function for evaluating the model
    Args:
        evaluation_circ: circuit in array form where evaluate the model
        evaluation_labels: labels of the noisy circuit 
        train_environment: environment used to train agent
        model: model to test 
    Return: 
        average reward (total reward/n_circuits), avg Hilbert-Schmidt distance, avg Trace Distance
    '''
    print(evaluation_circ.shape)
    avg_rew = []
    avg_trace_distance = []
    avg_bures_distance = []
    avg_fidelity = []
    correction = []
    n_circ=len(evaluation_circ)

    circuits=copy.deepcopy(evaluation_circ)
    
    environment = QuantumCircuit(
    circuits = circuits,
    representation = representation,
    labels = evaluation_labels,
    reward = reward, 
    )
    for i in range(n_circ):
        
        obs = environment.reset(i=i)
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            action = action[0]          
            obs, rewards, done, info = environment.step(action)
        predicted_circ = environment.get_qibo_circuit()
        dm_untrained = np.array(predicted_circ().state())
        #print(environment.get_circuit_rep()[:,:,4:].shape)
        correction.append(environment.get_circuit_rep()[:,:,4:])
        avg_rew.append(rewards)
        avg_fidelity.append(compute_fidelity(evaluation_labels[i],dm_untrained))
        avg_trace_distance.append(trace_distance(evaluation_labels[i],dm_untrained))
        avg_bures_distance.append(bures_distance(evaluation_labels[i],dm_untrained))

    rew = np.array(avg_rew)
    fid = np.array(avg_fidelity)
    trace_d = np.array(avg_trace_distance)
    bures_d = np.array(avg_bures_distance)
    correction = np.array(correction)
    results = np.array([(rew.mean(),rew.std(),
                       fid.mean(),fid.std(),
                       trace_d.mean(),trace_d.std(),
                       bures_d.mean(),bures_d.std(),
                       correction.mean(axis=0))],
                       dtype=[('reward','<f4'),('reward_std','<f4'),
                              ('fidelity','<f4'),('fidelity_std','<f4'),
                              ('trace_distance','<f4'),('trace_distance_std','<f4'),
                              ('bures_distance','<f4'),('bures_distance_std','<f4'),
                              ('avg_correction', np.float64, (evaluation_circ.shape[2],evaluation_circ.shape[1],4))                                                  
                            ])
    return results

def RB_evaluation(lambda_RB,circ_representation,target_label):
    dataset_size = len(target_label)
    trace_distance_rb_list = []
    bures_distance_rb_list = []
    fidelity_rb_list = []
    trace_distance_no_noise_list = []
    bures_distance_no_noise_list = []
    fidelity_no_noise_list = []
    rb_noise_model=CustomNoiseModel(["RX","RZ"],lam=lambda_RB,p0=0,epsilon_x=0,epsilon_z=0,
                               x_coherent_on_gate=["none"],z_coherent_on_gate=["none"],
                               depol_on_gate=["rx","rz"],damping_on_gate=["none"])
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

'''
def test_representation():
    print('> Noiseless Circuit:\n', circuit.draw())
    array = rep.circuit_to_array(circuit)
    print(' --> Representation:\n', array)
    print(' --> Circuit Rebuilt:\n', rep.array_to_circuit(array).draw())
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('> Noisy Circuit:\n', noisy_circuit.draw())
    array = rep.circuit_to_array(noisy_circuit)
    print(array)
    print(' --> Circuit Rebuilt:\n', rep.array_to_circuit(array).draw())
'''


from scipy.optimize import curve_fit
from qibo import gates

def randomized_benchmarking(circuits, backend=None, nshots=1000, noise_model=None):
    
    if backend is None:  # pragma: no cover
        from qibo.backends import GlobalBackend
        
        backend = GlobalBackend()
        
    nqubits = circuits[0].nqubits
    
    circ = { c.depth: [] for c in circuits }
    for c in circuits:
        depth = c.depth
        c.add(gates.Unitary(c.invert().unitary(), *range(nqubits)))
        if noise_model is not None:
            circ[depth].append(noise_model.apply(c))
        else:
            circ[depth].append(c)
        
    probs = { d: [] for d in circ.keys() }
    init_state = f"{0:0{nqubits}b}"
    for depth, circs in circ.items():
        for c in circs:
            c.add(gates.M(*range(nqubits)))
            freq = backend.execute_circuit(c, nshots=nshots).frequencies()
            if init_state not in freq:
                probs[depth].append(0)
            else:
                probs[depth].append(freq[init_state]/nshots)
    probs = [ (d, np.mean(p)) for d,p in probs.items() ]
    probs = sorted(probs, key=lambda x: x[0])
    model = lambda depth,a,l,b: a * np.power(l,depth) + b
    depths, survival_probs = zip(*probs)
    optimal_params, _ = curve_fit(model, depths, survival_probs, maxfev = 2000, p0=[1,0.5,0])
    model = lambda depth: optimal_params[0] * np.power(optimal_params[1],depth) + optimal_params[2]
    return depths, survival_probs, optimal_params, model


def fill_identity(circuit: Circuit):
    """Fill the circuit with identity gates where no gate is present to apply RB noisemodel.
    Works with circuits with no more than 3 qubits."""
    new_circuit = Circuit(circuit.nqubits)
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


class RL_NoiseModel(object):

    def __init__(self, agent, circuit_representation):
        super(self, ).__init__()
        self.agent = agent
        self.rep = circuit_representation

    def apply(self, circuit):
        if isinstance(circuit, Circuit):
            observation = self.rep.circuit_to_array(circuit)
        elif isinstance(circuit, np.ndarray):
            observation = circuit
        else:
            assert False, "Invalid circuit type"
        for i in range(circuit.shape[0]):
            action = self.agent.predict(observation, deterministic=True)
            observation = self.rep.make_action(action=action, circuit=observation, position=i)
        return self.rep.rep_to_circuit(observation)
            
            
if __name__ == "__main__":

    from rlnoise.dataset import Dataset, CircuitRepresentation
    from rlnoise.custom_noise import CustomNoiseModel
    import matplotlib.pyplot as plt

    nqubits=3
    noise_model = CustomNoiseModel()

    rep = CircuitRepresentation(
        primitive_gates = noise_model.primitive_gates,
        noise_channels = noise_model.channels,
        shape = '3d',
    )

    circs = []
    for depth in (2, 5, 10, 20, 30, 50):
        d = Dataset(20, depth, 3, rep, noise_model=noise_model)
        circs += d.circuits

    depths, survival_probs, optimal_params, model = randomized_benchmarking(circs, noise_model=noise_model)
    print(f"> Decay: {optimal_params[1]}")
    print(optimal_params)
    plt.scatter(depths, survival_probs)
    plt.plot(depths, model(depths))
    plt.show()
    

