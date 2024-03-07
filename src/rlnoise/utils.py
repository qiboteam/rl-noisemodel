import os
import json
import copy
import numpy as np
from pathlib import Path
from qibo import gates
from qibo.models.circuit import Circuit
import matplotlib.pyplot as plt
from rlnoise.custom_noise import CustomNoiseModel
from rlnoise.dataset import CircuitRepresentation
from rlnoise.rewards import DensityMatrixReward
from rlnoise.gym_env import QuantumCircuit
from qibo.transpiler.unroller import Unroller, NativeGates
from qibo.transpiler.optimizer import Rearrange
from rlnoise.metrics import trace_distance,bures_distance,compute_fidelity

with open(f"{Path(__file__).parent}/config.json") as f:
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

    circuits = copy.deepcopy(evaluation_circ)
    labels = copy.deepcopy(evaluation_labels)
    environment = QuantumCircuit(
    circuits = copy.deepcopy(circuits),
    representation = representation,
    labels = copy.deepcopy(labels),
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
        dm_untrained = predicted_circ().state()
        #print(environment.get_circuit_rep()[:,:,4:].shape)
        #correction.append(environment.get_circuit_rep()[:,:,4:])
        avg_rew.append(rewards)
        avg_fidelity.append(compute_fidelity(labels[i],dm_untrained))
        avg_trace_distance.append(trace_distance(labels[i],dm_untrained))
        avg_bures_distance.append(bures_distance(labels[i],dm_untrained))

    rew = np.array(avg_rew)
    fid = np.array(avg_fidelity)
    trace_d = np.array(avg_trace_distance)
    bures_d = np.array(avg_bures_distance)
    return np.array(
        [
            (
                rew.mean(),
                rew.std(),
                fid.mean(),
                fid.std(),
                trace_d.mean(),
                trace_d.std(),
                bures_d.mean(),
                bures_d.std(),
            )
        ],
        # correction.mean(axis=0))],
        dtype=[
            ('reward', '<f4'),
            ('reward_std', '<f4'),
            ('fidelity', '<f4'),
            ('fidelity_std', '<f4'),
            ('trace_distance', '<f4'),
            ('trace_distance_std', '<f4'),
            ('bures_distance', '<f4'),
            ('bures_distance_std', '<f4')
            # ('avg_correction', np.float64, (evaluation_circ.shape[2],evaluation_circ.shape[1],4))
        ],
    )

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


class RL_NoiseModel(object):

    def __init__(self, agent, circuit_representation, only_depol):
        self.agent = agent
        self.rep = circuit_representation
        self.ker_size = self.agent.policy.features_extractor._observation_space.shape[-1]
        self.ker_radius = int(self.ker_size/2)
        self.only_depolarizing = only_depol

    def apply(self, circuit):
        if isinstance(circuit, Circuit):
            circuit = self.rep.circuit_to_array(circuit)
        circuit = np.transpose(circuit, axes=[2,1,0])

        for pos in range(circuit.shape[-1]):
            l_flag = pos - self.ker_radius < 0
            r_flag = pos + self.ker_radius > circuit.shape[2] - 1
            if l_flag and not r_flag:
                pad_cel = np.zeros((circuit.shape[0], circuit.shape[1], np.abs(pos - self.ker_radius)))
                observation = np.concatenate((pad_cel, circuit[:, :, :pos + self.ker_radius + 1]), axis=2)
            elif not l_flag and r_flag:
                pad_cel = np.zeros((circuit.shape[0], circuit.shape[1], np.abs(pos + self.ker_radius - circuit.shape[-1] + 1)))
                observation = np.concatenate((circuit[:, :, pos - self.ker_radius:], pad_cel), axis=2)
            elif l_flag and r_flag:
                l_pad = np.zeros((circuit.shape[0], circuit.shape[1], np.abs(pos - self.ker_radius)))
                l_obs = np.concatenate((l_pad, circuit[:, :, :pos + self.ker_radius + 1]), axis=2)
                r_pad = np.zeros((circuit.shape[0], circuit.shape[1], np.abs(pos + self.ker_radius - circuit.shape[-1] + 1)))
                r_obs = np.concatenate((circuit[:, :, pos - self.ker_radius:], r_pad), axis=2)
                observation = np.concatenate((l_obs, r_obs), axis=2)
            else:
                observation = circuit[:, :, pos - self.ker_radius: pos + self.ker_radius + 1]   
            action, _ = self.agent.predict(observation, deterministic=True)
            circuit = self.rep.make_action(action=action, circuit=circuit, position=pos, only_depol = self.only_depolarizing)
        return self.rep.rep_to_circuit(np.transpose(circuit, axes=[2,1,0]))

def test_avg_fidelity(rho1,rho2):
    fidelity = []
    for i in range(len(rho1)):
        print(i, "fidelity: ", compute_fidelity(rho1[i],rho2[i]))
        fidelity.append(compute_fidelity(rho1[i],rho2[i]))
    avg_fidelity = np.array(fidelity).mean()
    return avg_fidelity

def u3_dec(gate):
    # t, p, l = gate.parameters
    params = gate.parameters
    t = params[0]
    p = params[1]
    l = params[2]
    #print("parameters", params)
    decomposition = []
    if l != 0.0:
        decomposition.append(gates.RZ(gate.qubits[0], l))
    decomposition.append(gates.RX(gate.qubits[0], np.pi/2, 0))
    if t != -np.pi:
        decomposition.append(gates.RZ(gate.qubits[0], t + np.pi))
    decomposition.append(gates.RX(gate.qubits[0], np.pi/2, 0))
    if p != -np.pi:
        decomposition.append(gates.RZ(gate.qubits[0], p + np.pi))
    return decomposition

def unroll_circuit(circuit):
    from qibo.transpiler.unitary_decompositions import u3_decomposition
    natives = NativeGates.U3 | NativeGates.CZ
    unroller = Unroller(native_gates = natives)
    optimizer = Rearrange()

    unrolled_circuit = unroller(circuit)
    print(unrolled_circuit.draw())
    opt_circuit = optimizer(unrolled_circuit)
    print(opt_circuit.draw())
    #opt_circuit = unrolled_circuit
    queue = opt_circuit.queue
    final_circuit = Circuit(circuit.nqubits, density_matrix=True)
    for gate in queue:
        if isinstance(gate, gates.CZ):
            final_circuit.add(gate)
        elif isinstance(gate, gates.RZ):
            final_circuit.add(gate)
        elif isinstance(gate, gates.Unitary):
            matrix = gate.matrix()
            u3_gate = gates.U3(gate.qubits[0], *u3_decomposition(matrix))
            decomposed = u3_dec(u3_gate)
            for decomposed_gate in decomposed:
                final_circuit.add(decomposed_gate)
        elif isinstance(gate, gates.U3):
            decomposed = u3_dec(gate)
            for decomposed_gate in decomposed:
                final_circuit.add(decomposed_gate)
    return final_circuit


def grover():
    """Creates a Grover circuit with 3 qubits.
    The circuit searches for the 11 state, the last qubit is ancillary"""
    circuit = Circuit(3, density_matrix=True)
    circuit.add(gates.RZ(0, np.pi/2))
    circuit.add(gates.RX(0, np.pi/2))
    circuit.add(gates.RZ(0, np.pi/2))
    circuit.add(gates.RZ(1, np.pi/2))
    circuit.add(gates.RX(1, np.pi/2))
    circuit.add(gates.RZ(1, np.pi/2))
    circuit.add(gates.RX(2, np.pi))
    circuit.add(gates.RZ(2, np.pi/2))
    circuit.add(gates.RX(2, np.pi/2))
    circuit.add(gates.RZ(2, np.pi/2))
    #Toffoli
    circuit.add(gates.CZ(1, 2))
    circuit.add(gates.RX(2, -np.pi / 4))
    circuit.add(gates.CZ(0, 2))
    circuit.add(gates.RX(2, np.pi / 4))
    circuit.add(gates.CZ(1, 2))
    circuit.add(gates.RX(2, -np.pi / 4))
    circuit.add(gates.CZ(0, 2))
    circuit.add(gates.RX(2, np.pi / 4))
    circuit.add(gates.RZ(1, np.pi / 4))
    circuit.add(gates.RZ(1, np.pi/2))
    circuit.add(gates.RX(1, np.pi/2))
    circuit.add(gates.RZ(1, np.pi/2))
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.RZ(0, np.pi / 4))
    circuit.add(gates.RX(1, -np.pi / 4))
    circuit.add(gates.CZ(0, 1))
    ###
    circuit.add(gates.RZ(0, np.pi/2))
    circuit.add(gates.RX(0, np.pi/2))
    circuit.add(gates.RZ(0, np.pi/2))
    circuit.add(gates.RX(0, np.pi))
    circuit.add(gates.RX(1, np.pi))
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.RX(0, np.pi))
    circuit.add(gates.RX(1, np.pi))
    circuit.add(gates.RZ(0, np.pi/2))
    circuit.add(gates.RX(0, np.pi/2))
    circuit.add(gates.RZ(0, np.pi/2))
    circuit.add(gates.RZ(1, np.pi/2))
    circuit.add(gates.RX(1, np.pi/2))
    circuit.add(gates.RZ(1, np.pi/2))
    return circuit

def qft():
    circuit = Circuit(3, density_matrix=True)
    circuit.add(gates.H(0))
    #Gate S
    circuit.add(gates.U3(1, 0, 0, np.pi / 4))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.U3(1, 0, 0, -np.pi / 4))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.RZ(0, np.pi / 4))
    #Gate T
    circuit.add(gates.U3(2, 0, 0, np.pi / 8))
    circuit.add(gates.CNOT(0, 2))
    circuit.add(gates.U3(2, 0, 0, -np.pi / 8))
    circuit.add(gates.CNOT(0, 2))
    circuit.add(gates.RZ(0, np.pi / 8))

    circuit.add(gates.H(1))
    #Gate S
    circuit.add(gates.U3(2, 0, 0, np.pi / 4))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.U3(2, 0, 0, -np.pi / 4))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.RZ(1, np.pi / 4))

    circuit.add(gates.H(2))
    
    return circuit