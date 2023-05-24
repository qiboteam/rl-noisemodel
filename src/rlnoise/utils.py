import os
import json
import copy
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from rlnoise.gym_env import QuantumCircuit
from rlnoise.custom_noise import CustomNoiseModel
from qibo.quantum_info import trace_distance
from scipy.linalg import sqrtm

config_path=str(Path().parent.absolute())+'/src/rlnoise/config.json'
with open(config_path) as f:
    config = json.load(f)

DEBUG=False

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

    sqrt_mat = sqrtm(density_matrix0).astype(dtype=np.complex64)
    sqrt_mat_sqrt = np.array(copy.deepcopy(sqrt_mat @ density_matrix1 @ sqrt_mat),dtype=np.complex64)

    evs = np.linalg.eigvalsh(sqrt_mat_sqrt)
    evs = np.real(evs)
    evs = np.where(evs > 0.0, evs, 0.0)
    trace = (np.sum(np.sqrt(evs))) ** 2

    return trace

def bures_distance(density_matrix0, density_matrix1):
    """ Compute the Bures distance between density matrices
    .. math::
        B( \rho , \sigma ) = -\sqrt{2*(1-sqrt(F(\sigma,\rho)))} where F is the fidelity
    """
    return np.sqrt(2*(1-np.sqrt(compute_fidelity(density_matrix0, density_matrix1))))



def model_evaluation(evaluation_circ,evaluation_labels,train_environment,model):
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

    avg_rew=[]
    avg_trace_distance=[]
    avg_bures_distance=[]
    avg_fidelity=[]
    n_circ=len(evaluation_circ)
    
    gym_env_params = config['gym_env']
    kernel_size = gym_env_params['kernel_size']
    step_reward = gym_env_params['step_reward']
    step_r_metric = gym_env_params['step_r_metric']
    neg_reward = gym_env_params['neg_reward']
    pos_reward = gym_env_params['pos_reward']
    action_penalty = gym_env_params['action_penalty']
    action_space = gym_env_params['action_space']

    circuits=copy.deepcopy(evaluation_circ)
    
    environment = QuantumCircuit(
    circuits = circuits,
    representation = train_environment.rep,
    labels = evaluation_labels,
    reward = train_environment.reward, 
    neg_reward=neg_reward,
    pos_reward=pos_reward,
    step_r_metric=step_r_metric,
    action_penality=action_penalty,
    action_space_type=action_space,
    kernel_size = kernel_size,
    step_reward=step_reward
    )

    for i in range(n_circ):
        
        obs = environment.reset(i=i)
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            action=action[0]          
            obs, rewards, done, info = environment.step(action)
        predicted_circ = environment.get_qibo_circuit()
        predicted_rep=environment.get_circuit_rep()
        dm_untrained=np.array(predicted_circ().state())
        avg_rew.append(rewards)
        avg_fidelity.append(compute_fidelity(evaluation_labels[i],dm_untrained))
        avg_trace_distance.append(trace_distance(evaluation_labels[i],dm_untrained))
        avg_bures_distance.append(bures_distance(evaluation_labels[i],dm_untrained))
        if i==0 and DEBUG:
            noise_model=CustomNoiseModel() 
            test_rep=evaluation_circ[i]
            test_circ=noise_model.apply(train_environment.rep.rep_to_circuit(test_rep))
            print('\nTrue noisy circuit')
            print(test_circ.draw())
            print('\nPredicted noisy circ: ')
            print(predicted_circ.draw())
            print("Predicted representation: \n", predicted_rep)
    rew=np.array(avg_rew)
    fid=np.array(avg_fidelity)
    trace_d=np.array(avg_trace_distance)
    bures_d=np.array(avg_bures_distance)
    results=np.array([(rew.mean(),rew.std(),
                       fid.mean(),fid.std(),
                       trace_d.mean(),trace_d.std(),
                       bures_d.mean(),bures_d.std())],
                       dtype=[('reward','<f4'),('reward_std','<f4'),
                              ('fidelity','<f4'),('fidelity_std','<f4'),
                              ('trace_distance','<f4'),('trace_distance_std','<f4'),
                              ('bures_distance','<f4'),('bures_distance_std','<f4')                         
                            ])
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


