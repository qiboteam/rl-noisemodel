
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

depths=np.arange(3,31,3)

with open('src/rlnoise/simulation_phase/RB/1Q/results/comparison2_results_1Q.npz',"rb") as f:
    tmp=np.load(f,allow_pickle=True)
    results_train=tmp['trained'].reshape(-1)
    results_rb = tmp['RB'].reshape(-1)
fidelity = {'model': np.array([results_train[i][2] for i in range(len(results_train))]),
            'RB': np.array([results_rb[i][0] for i in range(len(results_rb))]),
            'std_model': np.array([results_train[i][3] for i in range(len(results_train))]),
            'std_RB': np.array([results_rb[i][1] for i in range(len(results_rb))]),
            'no_noise': np.array([results_rb[i][6] for i in range(len(results_rb))]),
            'no_noise_std': np.array([results_rb[i][7] for i in range(len(results_rb))])
            }

trace_distance = {'model': np.array([results_train[i][4] for i in range(len(results_train))]),
                  'RB': np.array([results_rb[i][2] for i in range(len(results_rb))]),
                  'std_model': np.array([results_train[i][5] for i in range(len(results_train))]),
                  'std_RB': np.array([results_rb[i][3] for i in range(len(results_rb))]),
                  'no_noise': np.array([results_rb[i][8] for i in range(len(results_rb))]),
                  'no_noise_std': np.array([results_rb[i][9] for i in range(len(results_rb))])
                  }

bures_distance = {'model': np.array([results_train[i][5] for i in range(len(results_train))]),
                  'RB': np.array([results_rb[i][4] for i in range(len(results_rb))]),
                  'std_model': np.array([results_train[i][6] for i in range(len(results_train))]),
                  'std_RB': np.array([results_rb[i][5] for i in range(len(results_rb))]),
                  'no_noise': np.array([results_rb[i][10] for i in range(len(results_rb))]),
                  'no_noise_std': np.array([results_rb[i][11] for i in range(len(results_rb))])
                  }

fig=plt.figure()
# fig.suptitle('Train D=10,Val D= np.arange(3,31,3), len=50, Q=1, K=3, Coherent(e_z=0.1,e_x=0.05),Std_noise(lam=0.05,p0=0.05) ', fontsize=15)
ax=fig.add_subplot(111)
# ax1=fig.add_subplot(132)
# ax2=fig.add_subplot(133)
ax.plot(depths,fidelity['model'],marker='.',label='RL-Model',color='orange')
ax.fill_between(depths,fidelity['model']-fidelity['std_model'],fidelity['model']+fidelity['std_model'],alpha=0.2,color='orange')
ax.plot(depths,fidelity['RB'],marker='.',label='RB')
ax.fill_between(depths,fidelity['RB']-fidelity['std_RB'],fidelity['RB']+fidelity['std_RB'],alpha=0.2)
ax.plot(depths,fidelity['no_noise'],marker='.',label='w/o adding noise',color='green')
ax.fill_between(depths,fidelity['no_noise']-fidelity['no_noise_std'],fidelity['no_noise']+fidelity['no_noise_std'],alpha=0.2,color='green')
ax.legend()
# ax1.plot(depths,bures_distance['model'],marker='.',label='RL-Model',color='orange')
# ax1.fill_between(depths,bures_distance['model']-bures_distance['std_model'],bures_distance['model']+bures_distance['std_model'],alpha=0.2,color='orange')
# ax1.plot(depths,bures_distance['RB'],marker='.',label='RB')
# ax1.fill_between(depths,bures_distance['RB']-bures_distance['std_RB'],bures_distance['RB']+bures_distance['std_RB'],alpha=0.2)
# ax1.plot(depths,bures_distance['no_noise'],marker='.',label='w/o adding noise',color='green')
# ax1.fill_between(depths,bures_distance['no_noise']-bures_distance['no_noise_std'],bures_distance['no_noise']+bures_distance['no_noise_std'],alpha=0.2,color='green')
# ax1.legend()

# ax2.plot(depths,trace_distance['model'],marker='.',label='RL-Model',color='orange')
# ax2.fill_between(depths,trace_distance['model']-trace_distance['std_model'],trace_distance['model']+trace_distance['std_model'],alpha=0.2,color='orange')
# ax2.plot(depths,trace_distance['RB'],marker='.',label='RB')
# ax2.fill_between(depths,trace_distance['RB']-trace_distance['std_RB'],trace_distance['RB']+trace_distance['std_RB'],alpha=0.2)
# ax2.plot(depths,trace_distance['no_noise'],marker='.',label='w/o adding noise',color='green')
# ax2.fill_between(depths,trace_distance['no_noise']-trace_distance['no_noise_std'],trace_distance['no_noise']+trace_distance['no_noise_std'],alpha=0.2,color='green')
# ax2.legend()
# ax.set_ylim(0.8,1.1)
ax.set(xlabel='Circuit Depth', ylabel='Fidelity',xticks=depths,title="RL model benchmarking")
# ax1.set(xlabel='Circuit Depth', ylabel='Bures distance',title='RL model benchmarking',xticks=depths)
# ax2.set(xlabel='Circuit Depth', ylabel='Trace distance',xticks=depths)

# ax1.errorbar(depths,trace_distance['model'],yerr=trace_distance['std_model'],marker='x',label='RL-Model',color='orange',capsize=4)
# ax1.errorbar(depths,trace_distance['RB'],yerr=trace_distance['std_RB'],marker='x',label='RB',capsize=4)
# ax1.errorbar(depths,trace_distance['no_noise'],yerr=trace_distance['no_noise_std'],marker='x',label='w/o adding noise',capsize=4,color='green')
# ax1.set(xlabel='Circuit Depth', ylabel='Trace Distance',title='Average Trace distance between DM',xticks=depths)
# ax2.errorbar(depths,bures_distance['model'],yerr=bures_distance['std_model'],marker='x',label='RL-Model',color='orange',capsize=4)
# ax2.errorbar(depths,bures_distance['RB'],yerr=bures_distance['std_RB'],marker='x',label='RB',capsize=4)
# ax2.errorbar(depths,bures_distance['no_noise'],yerr=bures_distance['no_noise_std'],marker='x',label='w/o adding noise',capsize=4,color='green')
# ax2.set(xlabel='Circuit Depth', ylabel='Bures Distance',title='Average Bures distance between DM',xticks=depths)

plt.show()




# PLOTTING TRAINING RESULTS FOR DIFFERENT DATASET SIZE

# n_circ=[100,1000,10000]
# depth=7
# fig=plt.figure()
# ax=fig.add_subplot(111)
# for dataset_size in n_circ:
#     f = open(bench_results_path+"/test_size_D_%d_Dep-Term_CZ_3Q_%d.npz"%(depth,n_circ),"rb")
#     tmp=np.load(f,allow_pickle=True)     
#     results_train=tmp['train_results']
#     results_eval=tmp['val_results']
#     timesteps=tmp['timesteps']
#     f.close()
#     ax.plot(timesteps,results_train[:,0])

# plt.show()



# VIOLIN PLOTS FOR TRAINING RESULTS

# results_path = 'src/rlnoise/simulation_phase/1Q_training/train_results.npz'

# with open(results_path,"rb") as f:
#     tmp = np.load(f,allow_pickle=True)
#     time_steps = tmp['timesteps'].reshape(-1)
#     train_results = tmp['train_results']
#     eval_results = tmp['val_results']
#     train_fidelity = train_results['fidelity'].reshape(-1)
#     train_fidelity_std = train_results['fidelity_std'].reshape(-1)
#     eval_fidelity = eval_results['fidelity'].reshape(-1)
#     eval_fidelity_std = eval_results['fidelity_std'].reshape(-1)
#     train_trace_distance = train_results['trace_distance'].reshape(-1)
#     train_trace_distance_std = train_results['trace_distance_std'].reshape(-1)
#     eval_trace_distance = eval_results['trace_distance'].reshape(-1)
#     eval_trace_distance_std = eval_results['trace_distance_std'].reshape(-1)

# fig, ax = plt.subplots(1, 1, figsize=(15, 8))
# fig.suptitle('1 qubit', fontsize=15)

# ax[0].plot(time_steps, train_fidelity, marker='.')
# ax[0].plot(time_steps, eval_fidelity, marker='.')
# ax[0].fill_between(time_steps, train_fidelity - train_fidelity_std, 
#                    train_fidelity + train_fidelity_std, alpha=0.2)
# ax[0].fill_between(time_steps, eval_fidelity - eval_fidelity_std, 
#                    eval_fidelity + eval_fidelity_std, alpha=0.3)
# ax[0].set(xlabel='Episodes/1000', ylabel='Fidelity',title='1 qubit')
# ax[0].legend(['train_set', 'test_set'],loc='lower right')

# ax[1].plot(time_steps, train_trace_distance, marker='.')
# ax[1].plot(time_steps, eval_trace_distance, marker='.')
# ax[1].fill_between(time_steps, train_trace_distance - train_trace_distance_std, 
#                    train_trace_distance + train_trace_distance_std, alpha=0.2) 
# ax[1].fill_between(time_steps, eval_trace_distance - eval_trace_distance_std, 
#                    eval_trace_distance + eval_trace_distance_std, alpha=0.3)
# ax[1].set(xlabel='Timesteps/1000', ylabel='Trace Distance',title='Trace distance')
# ax[1].legend(['train_set', 'test_set'], loc='lower right')

# plt.subplots_adjust(top=0.707)
# plt.show()