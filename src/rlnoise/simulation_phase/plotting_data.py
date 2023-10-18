
import matplotlib.pyplot as plt
import numpy as np
import os
'''
n_circ=[100,1000,10000]
depth=7
fig=plt.figure()
ax=fig.add_subplot(111)
for dataset_size in n_circ:
    f = open(bench_results_path+"/test_size_D_%d_Dep-Term_CZ_3Q_%d.npz"%(depth,n_circ),"rb")
    tmp=np.load(f,allow_pickle=True)     
    results_train=tmp['train_results']
    results_eval=tmp['val_results']
    timesteps=tmp['timesteps']
    f.close()
    ax.plot(timesteps,results_train[:,0])

plt.show()

'''

# VIOLIN PLOTS

results_path = 'src/rlnoise/data_analysis/simulation_phase/training_analysis/1Q/train_results.npz'

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

fig, ax = plt.subplots(1, 2, figsize=(15, 8))
fig.suptitle('subtitle', fontsize=15)

ax[0].plot(time_steps, train_fidelity, marker='.')
ax[0].plot(time_steps, eval_fidelity, marker='.')
ax[0].fill_between(time_steps, train_fidelity - train_fidelity_std, 
                   train_fidelity + train_fidelity_std, alpha=0.2)
ax[0].fill_between(time_steps, eval_fidelity - eval_fidelity_std, 
                   eval_fidelity + eval_fidelity_std, alpha=0.3)
ax[0].set(xlabel='timesteps', ylabel='Fidelity',title='Fidelity between DM')
ax[0].legend(['train_set', 'test_set'])

ax[1].plot(time_steps, train_trace_distance, marker='.')
ax[1].plot(time_steps, eval_trace_distance, marker='.')
ax[1].fill_between(time_steps, train_trace_distance - train_trace_distance_std, 
                   train_trace_distance + train_trace_distance_std, alpha=0.) 
ax[1].fill_between(time_steps, eval_trace_distance - eval_trace_distance_std, 
                   eval_trace_distance + eval_trace_distance_std, alpha=0.3)
ax[1].set(xlabel='timesteps', ylabel='Trace Distance',title='Trace distance between DM')
ax[1].legend(['train_set', 'test_set'])

plt.subplots_adjust(top=0.707)
plt.show()