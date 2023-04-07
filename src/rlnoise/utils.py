import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import os
from rlnoise.gym_env import QuantumCircuit
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

def model_evaluation(evaluation_circ,evaluation_labels,train_environment,model):
    '''
evaluation_circ: circuit in array form where evaluate the model
evaluation_labels: labels of the noisy circuit 
train_environment: environment used to train agent
model: model to test \n
return: average reward (total reward/n_circuits)
    '''
    environment = QuantumCircuit(
    circuits = evaluation_circ,
    representation = train_environment.rep,
    labels = evaluation_labels,
    reward = train_environment.reward,
    kernel_size=train_environment.kernel_size   
    )
    avg_rew=0
    n_circ=len(evaluation_circ)
    for i in range(n_circ):
        obs = environment.reset(i=i)
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = environment.step(action)
        untrained_circ = environment.get_qibo_circuit()
        dm_untrained=np.array(untrained_circ().state())
        avg_rew += rewards

    return avg_rew/n_circ

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