import matplotlib.pyplot as plt
import numpy as np
import os

SMALL_SIZE = 15
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

results_path = 'src/rlnoise/simulation_phase/1Q_training_new/train_results.npz'

with open(results_path,"rb") as f:
    tmp = np.load(f,allow_pickle=True)
    time_steps = tmp['timesteps'].reshape(-1)
    train_results = tmp['train_results']
    eval_results = tmp['val_results']
    train_fidelity = train_results['fidelity'].reshape(-1)
    train_fidelity_std = train_results['fidelity_std'].reshape(-1)
    eval_fidelity = eval_results['fidelity'].reshape(-1)
    eval_fidelity_std = eval_results['fidelity_std'].reshape(-1)
    train_trace_distance = train_results['trace_distance'].reshape(-1)
    train_trace_distance_std = train_results['trace_distance_std'].reshape(-1)
    eval_trace_distance = eval_results['trace_distance'].reshape(-1)
    eval_trace_distance_std = eval_results['trace_distance_std'].reshape(-1)

fig, ax = plt.subplots(1, 1, figsize=(15, 8))
fig.suptitle('1 qubit', fontsize=15)

ax[0].plot(time_steps, train_fidelity, marker='.')
ax[0].plot(time_steps, eval_fidelity, marker='.')
ax[0].fill_between(time_steps, train_fidelity - train_fidelity_std, 
                   train_fidelity + train_fidelity_std, alpha=0.2)
ax[0].fill_between(time_steps, eval_fidelity - eval_fidelity_std, 
                   eval_fidelity + eval_fidelity_std, alpha=0.3)
ax[0].set(xlabel='Episodes/1000', ylabel='Fidelity',title='1 qubit')
ax[0].legend(['train_set', 'test_set'],loc='lower right')

ax[1].plot(time_steps, train_trace_distance, marker='.')
ax[1].plot(time_steps, eval_trace_distance, marker='.')
ax[1].fill_between(time_steps, train_trace_distance - train_trace_distance_std, 
                   train_trace_distance + train_trace_distance_std, alpha=0.2) 
ax[1].fill_between(time_steps, eval_trace_distance - eval_trace_distance_std, 
                   eval_trace_distance + eval_trace_distance_std, alpha=0.3)
ax[1].set(xlabel='Timesteps/1000', ylabel='Trace Distance',title='Trace distance')
ax[1].legend(['train_set', 'test_set'], loc='lower right')

plt.subplots_adjust(top=0.707)
plt.show()