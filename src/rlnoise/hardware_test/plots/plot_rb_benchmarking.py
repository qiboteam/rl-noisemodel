
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use('science')
import scienceplots

plt.style.use('science')

qubits = 1

SMALL_SIZE = 22
MEDIUM_SIZE = 26
BIGGER_SIZE = 28

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

depths=np.arange(3,31,3)

if qubits == 1:
    filepath = 'src/rlnoise/simulation_phase/RB/1Q/results/comparison2_results_1Q.npz'
else:
    filepath = 'src/rlnoise/simulation_phase/RB/3Q/results/3q.npz'

with open(filepath,"rb") as f:
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

fig=plt.figure(figsize=(12, 9))
ax=fig.add_subplot(111)
ax.plot(depths,fidelity['model'],label='RL-Model', linewidth=4, color='#e60049')
ax.fill_between(depths,fidelity['model']-fidelity['std_model'],fidelity['model']+fidelity['std_model'],alpha=0.2, color='#e60049')
ax.plot(depths,fidelity['RB'],label='RB', linewidth=4, color='#0bb4ff')
ax.fill_between(depths,fidelity['RB']-fidelity['std_RB'],fidelity['RB']+fidelity['std_RB'],alpha=0.2, color='#0bb4ff')
ax.plot(depths,fidelity['no_noise'], label='Noiseless',linewidth=4, color='green')
ax.fill_between(depths,fidelity['no_noise']-fidelity['no_noise_std'],fidelity['no_noise']+fidelity['no_noise_std'],alpha=0.2,color='green')
ax.legend()
ax.set(xlabel='Circuit Depth', ylabel='Fidelity', xticks=depths)

plt.savefig(f"{qubits}Q_rb.pdf", )
plt.show()

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