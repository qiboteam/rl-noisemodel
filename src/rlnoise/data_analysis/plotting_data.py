
import matplotlib.pyplot as plt
import numpy as np
import os
bench_results_path=os.getcwd()+'/src/rlnoise/bench_results'

                                                    #DIFFERENT TRAIN AND VALID DEPTH

depths=[5,7,10,15,30]
                                                
#f = open(bench_results_path+"/AllNoise_len1000_1Msteps"+str(depths),"rb")
#tmp=np.load(f,allow_pickle=True)     

#results_train1=tmp['trained']

#f.close()
f = open(bench_results_path+"/AllNoise_len1000_1Msteps"+str(depths),"rb")
tmp=np.load(f,allow_pickle=True)     
results_train=tmp['trained'].reshape(-1)
results_untrain=tmp['untrained'].reshape(-1)
f.close()

fig=plt.figure()
fig.suptitle('Train D=5,Val D=[5,7,10,15,30], len=1000, Q=1, K=3, Coherent(e_z=0.1,e_x=0.2),Std_noise(lam=0.05,p0=0.1) ', fontsize=15)
ax=fig.add_subplot(131)
ax1=fig.add_subplot(132)
ax2=fig.add_subplot(133)


ax.errorbar(depths,results_train["reward"],yerr=results_train["reward_std"] ,marker='x',label='Trained',color='orange',capsize=4)
ax.errorbar(depths,results_untrain["reward"],yerr=results_train["reward_std"],marker='x',label='Untrained',capsize=4)
#ax.plot(depths,results_train1[:,0],marker='x',label='Slightly changed lam',color='red')
ax.legend()
ax.set(xlabel='Circuit Depth', ylabel='Reward',title='Average final reward',xticks=depths)
ax1.errorbar(depths,results_train["fidelity"],yerr=results_train["fidelity_std"],marker='x',label='Trained',color='orange',capsize=4)
ax1.errorbar(depths,results_untrain["fidelity"],yerr=results_train["fidelity_std"],marker='x',label='Untrained',capsize=4)
#ax1.plot(depths,results_train1[:,1],marker='x',label='Untrained',color='red')
ax1.set(xlabel='Circuit Depth', ylabel='Fidelity',title='Fidelity between DM',xticks=depths)
ax2.errorbar(depths,results_train["trace_distance"],yerr=results_train["trace_distance_std"],marker='x',label='Trained',color='orange',capsize=4)
ax2.errorbar(depths,results_untrain["trace_distance"],yerr=results_train["trace_distance_std"],marker='x',label='Untrained',capsize=4)
#ax2.plot(depths,results_train1[:,2],marker='x',label='Untrained',color='red')
ax2.set(xlabel='Circuit Depth', ylabel='Trace Distance',title='Average Trace distance between DM',xticks=depths)


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