
import matplotlib.pyplot as plt
import numpy as np
import os
bench_results_path=os.getcwd()+'/src/rlnoise/bench_results'
time_steps=[2000,4000,6000,8000,10000,15000,20000,30000,50000,80000,130000,200000]

                                                    #TRAIN VS UNTRAINED PLOTS MULTIPLE TIMESTEPS
                                             
f = open(bench_results_path+"/Dep-Term_CZ_3Q"+str(time_steps),"rb")
tmp=np.load(f,allow_pickle=True)

untrained_results=tmp['untrained']
trained_results=tmp['trained']
f.close()
fig=plt.figure()
fig.suptitle('Train and validation depth=7, 3 qubit w CZ, Depolarizing and Thermal (fixed), Step_reward=False, Kernel_size=3 ', fontsize=15)
ax=fig.add_subplot(131)
ax1=fig.add_subplot(132)
ax2=fig.add_subplot(133)


ax.plot(time_steps,untrained_results[:,0],label='Untrained',marker='x')
ax.set(xlabel='total_timesteps', ylabel='Reward',title='Average final reward')
ax1.plot(time_steps,untrained_results[:,1],marker='x')
ax1.set(xlabel='total_timesteps', ylabel='MAE',title='Average MAE between density matrices')
ax2.plot(time_steps,untrained_results[:,2],marker='x')
ax2.set(xlabel='total_timesteps', ylabel='Trace Distance',title='Average trace distance between density matrices')

ax.plot(time_steps,trained_results[:,0],color='orange',label='Trained',marker='x')

ax1.plot(time_steps,trained_results[:,1],color='orange',label='Trained',marker='x')

ax2.plot(time_steps,trained_results[:,2],color='orange',label='Trained',marker='x')
ax.legend()

plt.show()


                                                    #DIFFERENT TRAIN AND VALID DEPTH
'''
depths=[7,10,20,30]
                                                
f = open(bench_results_path+"/Dep-Term_CZ_3Q_100k"+str(depths),"rb")
tmp=np.load(f,allow_pickle=True)     

results_train=tmp['trained']
results_untrain=tmp['untrained']
f.close()

fig=plt.figure()
fig.suptitle('Train depth=7 val depths=[7,10,20,30],3 qubit w CZ, Depol and Thermal (fixed), Step_reward=False, Kernel_size=3 ', fontsize=15)
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
'''