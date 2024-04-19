import matplotlib.pyplot as plt
import numpy as np
import scienceplots
plt.style.use('science')

results_path = "simulation/experiments/3q_high_noise/model_train_result.npz"
steps = 120
mse = True

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

if not mse:
    fig=plt.figure(figsize=(12, 9))
    ax=fig.add_subplot(111)

    ax.plot(time_steps, train_fidelity, linewidth=4, color='#e60049')
    ax.plot(time_steps, eval_fidelity, linewidth=4, color='#0bb4ff')
    ax.fill_between(time_steps, train_fidelity - train_fidelity_std, 
                    train_fidelity + train_fidelity_std, alpha=0.2, color='#e60049')
    ax.fill_between(time_steps, eval_fidelity - eval_fidelity_std, 
                    eval_fidelity + eval_fidelity_std, alpha=0.2, color='#0bb4ff')
    ax.set(xlabel='Episodes/1000', ylabel='Fidelity')
    ax.legend(['Train Set', 'Test Set'],loc='lower right')

else:
    fig, ax = plt.subplots(1, 2, figsize=(25, 9))

    ax[0].plot(time_steps, train_fidelity, linewidth=4, color='#e60049')
    ax[0].plot(time_steps, eval_fidelity, linewidth=4, color='#0bb4ff')
    ax[0].fill_between(time_steps, train_fidelity - train_fidelity_std, 
                    train_fidelity + train_fidelity_std, alpha=0.2, color='#e60049')
    ax[0].fill_between(time_steps, eval_fidelity - eval_fidelity_std, 
                    eval_fidelity + eval_fidelity_std, alpha=0.2, color='#0bb4ff')
    ax[0].set(xlabel='Episodes/1000', ylabel='Fidelity')
    ax[0].legend(['Train Set', 'Test Set'], loc='lower right')

    ax[1].plot(time_steps, train_mse, linewidth=4, color='#e60049')
    ax[1].plot(time_steps, eval_mse, linewidth=4, color='#0bb4ff')
    ax[1].fill_between(time_steps, train_mse - train_mse_std, 
                    train_mse + train_mse_std, alpha=0.2, color='#e60049')
    ax[1].fill_between(time_steps, eval_mse - eval_mse_std, 
                    eval_mse + eval_mse_std, alpha=0.2, color='#0bb4ff')
    ax[1].set(xlabel='Episodes/1000', ylabel='MSE')
    ax[1].legend(['Train Set', 'Test Set'], loc='lower right')

plt.savefig(f"Train_results.pdf")
plt.show()