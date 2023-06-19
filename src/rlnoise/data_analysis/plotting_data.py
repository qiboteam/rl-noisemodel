
import matplotlib.pyplot as plt
import numpy as np
import os
bench_results_path=os.getcwd()+'/src/rlnoise/bench_results/'

                                                    #DIFFERENT TRAIN AND VALID DEPTH

depths=np.arange(3,31,3)
                                                
f = open(bench_results_path+"AllNoise_len50_1Msteps_1Q.npz","rb")
tmp=np.load(f,allow_pickle=True)     
results_train=tmp['trained'].reshape(-1)
results_rb = tmp['RB'].reshape(-1)
f.close()

fidelity = {'model': [results_train[i][2] for i in range(len(results_train))],
            'RB': [results_rb[i][0] for i in range(len(results_rb))],
            'std_model': [results_train[i][3] for i in range(len(results_train))],
            'std_RB': [results_rb[i][1] for i in range(len(results_rb))],
            'no_noise': [results_rb[i][6] for i in range(len(results_rb))],
            'no_noise_std': [results_rb[i][7] for i in range(len(results_rb))]}

trace_distance = {'model': [results_train[i][4] for i in range(len(results_train))],
                  'RB': [results_rb[i][2] for i in range(len(results_rb))],
                  'std_model': [results_train[i][5] for i in range(len(results_train))],
                  'std_RB': [results_rb[i][3] for i in range(len(results_rb))],
                  'no_noise': [results_rb[i][8] for i in range(len(results_rb))],
                  'no_noise_std': [results_rb[i][9] for i in range(len(results_rb))]}

bures_distance = {'model': [results_train[i][5] for i in range(len(results_train))],
                  'RB': [results_rb[i][4] for i in range(len(results_rb))],
                  'std_model': [results_train[i][6] for i in range(len(results_train))],
                  'std_RB': [results_rb[i][5] for i in range(len(results_rb))],
                  'no_noise': [results_rb[i][10] for i in range(len(results_rb))],
                  'no_noise_std': [results_rb[i][11] for i in range(len(results_rb))]}

fig=plt.figure()
fig.suptitle('Train D=10,Val D= np.arange(3,31,3), len=50, Q=1, K=3, Coherent(e_z=0.1,e_x=0.05),Std_noise(lam=0.05,p0=0.05) ', fontsize=15)
ax=fig.add_subplot(131)
ax1=fig.add_subplot(132)
ax2=fig.add_subplot(133)
ax.errorbar(depths,fidelity['model'],yerr=fidelity['std_model'],marker='x',label='RL-Model',color='orange',capsize=4)
ax.errorbar(depths,fidelity['RB'],yerr=fidelity['std_RB'],marker='x',label='RB',capsize=4)
ax.errorbar(depths,fidelity['no_noise'],yerr=fidelity['no_noise_std'],marker='x',label='w/o adding noise',capsize=4,color='green')
ax.legend()
ax.set(xlabel='Circuit Depth', ylabel='Fidelity',title='Fidelity between DM',xticks=depths)
ax1.errorbar(depths,trace_distance['model'],yerr=trace_distance['std_model'],marker='x',label='RL-Model',color='orange',capsize=4)
ax1.errorbar(depths,trace_distance['RB'],yerr=trace_distance['std_RB'],marker='x',label='RB',capsize=4)
ax1.errorbar(depths,trace_distance['no_noise'],yerr=trace_distance['no_noise_std'],marker='x',label='w/o adding noise',capsize=4,color='green')
ax1.set(xlabel='Circuit Depth', ylabel='Trace Distance',title='Average Trace distance between DM',xticks=depths)
ax2.errorbar(depths,bures_distance['model'],yerr=bures_distance['std_model'],marker='x',label='RL-Model',color='orange',capsize=4)
ax2.errorbar(depths,bures_distance['RB'],yerr=bures_distance['std_RB'],marker='x',label='RB',capsize=4)
ax2.errorbar(depths,bures_distance['no_noise'],yerr=bures_distance['no_noise_std'],marker='x',label='w/o adding noise',capsize=4,color='green')
ax2.set(xlabel='Circuit Depth', ylabel='Bures Distance',title='Average Bures distance between DM',xticks=depths)

plt.show()


                            #SAME SETUP BUT DIFFERENT DATASET SIZE (n_circ)
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