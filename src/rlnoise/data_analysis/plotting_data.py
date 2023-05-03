
import matplotlib.pyplot as plt
import numpy as np
import os
bench_results_path=os.getcwd()+'/src/rlnoise/bench_results'
time_steps=[10000,15000,20000,25000,30000,40000]

                                                    #TRAIN VS UNTRAINED PLOTS
'''                                                    
f = open(bench_results_path+"/Depol_only_D5_Q1_K3_SR-off_ts"+str(time_steps),"rb")
tmp=np.load(f,allow_pickle=True)

untrained_results=tmp['untrained']
trained_results=tmp['trained']
f.close()
fig=plt.figure()
fig.suptitle('Train and validation depth=5, 1 qubit, Depolarizing only (fixed lambda), Step_reward=False, Kernel_size=3 ', fontsize=15)
ax=fig.add_subplot(131)
ax1=fig.add_subplot(132)
ax2=fig.add_subplot(133)


ax.plot(time_steps,untrained_results[:,0],label='Untrained')
ax.set(xlabel='total_timesteps', ylabel='Reward',title='Average final reward')
ax1.plot(time_steps,untrained_results[:,1])
ax1.set(xlabel='total_timesteps', ylabel='MAE',title='Average MAE between density matrices')
ax2.plot(time_steps,untrained_results[:,2])
ax2.set(xlabel='total_timesteps', ylabel='Trace Distance',title='Average trace distance between density matrices')

ax.plot(time_steps,trained_results[:,0],color='orange',label='Trained')

ax1.plot(time_steps,trained_results[:,1],color='orange',label='Trained')

ax2.plot(time_steps,trained_results[:,2],color='orange',label='Trained')
ax.legend()

plt.show()
'''

                                                    #DIFFERENT TRAIN AND VALID DEPTH
depths=[5,10,15,20]                                                   
f = open(bench_results_path+"/Depol_only_D5(train)_Q1_K3_SR-off_Test"+str(depths),"rb")
tmp=np.load(f,allow_pickle=True)     

results_train=tmp['trained']
results_untrain=tmp['untrained']
f.close()

fig=plt.figure()
fig.suptitle('Train depth=5 validation depth=[5,10,15,20], 1 qubit, Depolarizing only (fixed lambda), Step_reward=False, Kernel_size=3 ', fontsize=15)
ax=fig.add_subplot(131)
ax1=fig.add_subplot(132)
ax2=fig.add_subplot(133)


ax.plot(depths,results_train[:,0],marker='x',label='Trained',color='orange')
ax.plot(depths,results_untrain[:,0],marker='x',label='Untrained')
ax.legend()
ax.set(xlabel='Circuit Depth', ylabel='Reward',title='Average final reward',xticks=depths)
ax1.plot(depths,results_train[:,1],marker='x',label='Trained',color='orange')
ax1.plot(depths,results_untrain[:,1],marker='x',label='Untrained')
ax1.set(xlabel='Circuit Depth', ylabel='MAE',title='Average MAE between density matrices',xticks=depths)
ax2.plot(depths,results_train[:,2],marker='x',label='Trained',color='orange')
ax2.plot(depths,results_untrain[:,2],marker='x',label='Untrained')
ax2.set(xlabel='Circuit Depth', ylabel='Trace Distance',title='Average trace distance between density matrices',xticks=depths)


plt.show()