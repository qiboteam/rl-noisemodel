import matplotlib.pyplot as plt
import numpy as np
import scienceplots
plt.style.use('science')

exp_folder = "hardware/experiments/qw11qD4/"
results_path = exp_folder + "model_train_result.npz"
steps = 100

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

with open(results_path,"rb") as f:
    tmp = np.load(f,allow_pickle=True)
    time_steps = tmp['timesteps'].reshape(-1)[0:steps]
    train_results = tmp['train_results'][0:steps]
    eval_results = tmp['val_results'][0:steps]
    train_fidelity = train_results['fidelity'].reshape(-1)
    train_fidelity_std = train_results['fidelity_std'].reshape(-1)
    eval_fidelity = eval_results['fidelity'].reshape(-1)
    eval_fidelity_std = eval_results['fidelity_std'].reshape(-1)
    train_mse = train_results['mse'].reshape(-1)
    train_mse_std = train_results['mse_std'].reshape(-1)
    eval_mse = eval_results['mse'].reshape(-1)
    eval_mse_std = eval_results['mse_std'].reshape(-1)
    train_reward = train_results['reward'].reshape(-1)
    train_reward_std = train_results['reward_std'].reshape(-1)
    eval_reward = eval_results['reward'].reshape(-1)
    eval_reward_std = eval_results['reward_std'].reshape(-1)
    trace_distance = train_results['trace_distance'].reshape(-1)
    trace_distance_std = train_results['trace_distance_std'].reshape(-1)
    eval_trace_distance = eval_results['trace_distance'].reshape(-1)
    eval_trace_distance_std = eval_results['trace_distance_std'].reshape(-1)

fig, ax = plt.subplots(2, 2, figsize=(30, 20))

ax[0,0].plot(time_steps, train_fidelity, linewidth=4, color='#e60049')
ax[0,0].plot(time_steps, eval_fidelity, linewidth=4, color='#0bb4ff')
ax[0,0].fill_between(time_steps, train_fidelity - train_fidelity_std, 
                train_fidelity + train_fidelity_std, alpha=0.2, color='#e60049')
ax[0,0].fill_between(time_steps, eval_fidelity - eval_fidelity_std, 
                eval_fidelity + eval_fidelity_std, alpha=0.2, color='#0bb4ff')
ax[0,0].set(xlabel='Episodes/1000', ylabel='Fidelity')
ax[0,0].legend(['Train Set', 'Test Set'], loc='lower right')

ax[0,1].plot(time_steps, train_mse, linewidth=4, color='#e60049')
ax[0,1].plot(time_steps, eval_mse, linewidth=4, color='#0bb4ff')
ax[0,1].fill_between(time_steps, train_mse - train_mse_std, 
                train_mse + train_mse_std, alpha=0.2, color='#e60049')
ax[0,1].fill_between(time_steps, eval_mse - eval_mse_std, 
                eval_mse + eval_mse_std, alpha=0.2, color='#0bb4ff')
ax[0,1].set(xlabel='Episodes/1000', ylabel='MSE')
ax[0,1].legend(['Train Set', 'Test Set'], loc='lower right')

ax[1,0].plot(time_steps, train_reward, linewidth=4, color='#e60049')
ax[1,0].plot(time_steps, eval_reward, linewidth=4, color='#0bb4ff')
ax[1,0].fill_between(time_steps, train_reward - train_reward_std, 
                train_reward + train_reward_std, alpha=0.2, color='#e60049')
ax[1,0].fill_between(time_steps, eval_reward - eval_reward_std,
                eval_reward + eval_reward_std, alpha=0.2, color='#0bb4ff')
ax[1,0].set(xlabel='Episodes/1000', ylabel='Reward')
ax[1,0].legend(['Train Set', 'Test Set'], loc='lower right')

ax[1,1].plot(time_steps, trace_distance, linewidth=4, color='#e60049')
ax[1,1].plot(time_steps, eval_trace_distance, linewidth=4, color='#0bb4ff')
ax[1,1].fill_between(time_steps, trace_distance - trace_distance_std, 
                trace_distance + trace_distance_std, alpha=0.2, color='#e60049')
ax[1,1].fill_between(time_steps, eval_trace_distance - eval_trace_distance_std,
                eval_trace_distance + eval_trace_distance_std, alpha=0.2, color='#0bb4ff')
ax[1,1].set(xlabel='Episodes/1000', ylabel='Trace Distance')
ax[1,1].legend(['Train Set', 'Test Set'], loc='upper right')

plt.savefig(exp_folder + "images/Train_results.pdf")
plt.show()