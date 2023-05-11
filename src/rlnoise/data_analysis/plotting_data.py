
import matplotlib.pyplot as plt
import numpy as np
import os
bench_results_path=os.getcwd()+'/src/rlnoise/bench_results'

                                                    #DIFFERENT TRAIN AND VALID DEPTH
'''
depths=[7,10,15,20,25,30,35,40]
                                                
f = open(bench_results_path+"/Dep-Term_CZ_3Q_154k_1"+str(depths),"rb")
tmp=np.load(f,allow_pickle=True)     

results_train1=tmp['trained']

f.close()
f = open(bench_results_path+"/Dep-Term_CZ_3Q_154k"+str(depths),"rb")
tmp=np.load(f,allow_pickle=True)     

results_train=tmp['trained']
results_untrain=tmp['untrained']
f.close()
fig=plt.figure()
fig.suptitle('Train D=7,Val D=[7,10,15,20,25,30,35,40],3 qubit w CZ, T(0.05/0.07) Valid(0.1/0.07), Step_reward=False, Kernel_size=3 ', fontsize=15)
ax=fig.add_subplot(131)
ax1=fig.add_subplot(132)
ax2=fig.add_subplot(133)


ax.plot(depths,results_train[:,0],marker='x',label='Trained',color='orange')
ax.plot(depths,results_untrain[:,0],marker='x',label='Untrained')
ax.plot(depths,results_train1[:,0],marker='x',label='Slightly changed lam',color='red')
ax.legend()
ax.set(xlabel='Circuit Depth', ylabel='Reward',title='Average final reward',xticks=depths)
ax1.plot(depths,results_train[:,1],marker='x',label='Trained',color='orange')
ax1.plot(depths,results_untrain[:,1],marker='x',label='Untrained')
ax1.plot(depths,results_train1[:,1],marker='x',label='Untrained',color='red')
ax1.set(xlabel='Circuit Depth', ylabel='H-S',title='Average Hilbert-Schmidt distance between DM',xticks=depths)
ax2.plot(depths,results_train[:,2],marker='x',label='Trained',color='orange')
ax2.plot(depths,results_untrain[:,2],marker='x',label='Untrained')
ax2.plot(depths,results_train1[:,2],marker='x',label='Untrained',color='red')
ax2.set(xlabel='Circuit Depth', ylabel='Trace Distance',title='Average Trace distance between DM',xticks=depths)


plt.show()
'''

                            #SAME SETUP BUT DIFFERENT DATASET SIZE (n_circ)

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

