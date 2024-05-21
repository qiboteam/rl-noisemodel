import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use('science')

mse_plot = True
exp_folder = "simulation/experiments/3q_squared/"
rb_result = exp_folder + "rb_result.npz"
rl_result = exp_folder + "rl_result.npz"

fidelity = {}
mse = {}
trace = {}
with open(rb_result, "rb") as f:
    rb_result = np.load(f, allow_pickle=True)
    rb_result = rb_result['result']
    depths = rb_result['depth'].flatten()
    fidelity['RB'] = rb_result['fidelity'].flatten()
    fidelity['std_RB'] = rb_result['fidelity_std'].flatten()
    fidelity['no_noise'] = rb_result['fidelity_no_noise'].flatten()
    fidelity['no_noise_std'] = rb_result['fidelity_no_noise_std'].flatten()
    fidelity['mms'] = rb_result['fidelity_mms'].flatten()
    fidelity['mms_std'] = rb_result['fidelity_mms_std'].flatten()
    mse["RB"] = rb_result['mse'].flatten()
    mse["std_RB"] = rb_result['mse_std'].flatten()
    mse["no_noise"] = rb_result['mse_no_noise'].flatten()
    mse["no_noise_std"] = rb_result['mse_no_noise_std'].flatten()
    mse['mms'] = rb_result['mse_mms'].flatten()
    mse['mms_std'] = rb_result['mse_mms_std'].flatten()
    trace['RB'] = rb_result['trace_distance'].flatten()
    trace['std_RB'] = rb_result['trace_distance_std'].flatten()
    trace['no_noise'] = rb_result['trace_distance_no_noise'].flatten()
    trace['no_noise_std'] = rb_result['trace_distance_no_noise_std'].flatten()
    trace['mms'] = rb_result['trace_distance_mms'].flatten()
    trace['mms_std'] = rb_result['trace_distance_mms_std'].flatten()

with open(rl_result, "rb") as f:
    rl_result = np.load(f, allow_pickle=True)
    rl_result=rl_result['result']
    fidelity['model'] = rl_result['fidelity'].flatten()
    fidelity['std_model'] = rl_result['fidelity_std'].flatten()
    mse['model'] = rl_result['mse'].flatten()
    mse['std_model'] = rl_result['mse_std'].flatten()
    trace['model'] = rl_result['trace_distance'].flatten()
    trace['std_model'] = rl_result['trace_distance_std'].flatten()

SMALL_SIZE = 24
MEDIUM_SIZE = 26
BIGGER_SIZE = 28

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

if not mse_plot:
    fig=plt.figure(figsize=(25,8))
    ax=fig.add_subplot(111)

    ax.plot(depths, fidelity['model'], label = 'RL-Model', linewidth=4, color='#e60049')
    ax.fill_between(depths, fidelity['model'] - fidelity['std_model'], fidelity['model'] + fidelity['std_model'], alpha=0.2, color='#e60049')
    ax.plot(depths, fidelity['RB'], label='RB', linewidth = 4, color='#0bb4ff')
    ax.fill_between(depths, fidelity['RB'] - fidelity['std_RB'], fidelity['RB'] + fidelity['std_RB'], alpha=0.2, color='#0bb4ff')
    ax.plot(depths, fidelity['no_noise'], label = 'No noise added', linewidth = 4, color='green')
    ax.fill_between(depths, fidelity['no_noise'] - fidelity['no_noise_std'], fidelity['no_noise'] + fidelity['no_noise_std'], alpha=0.2, color='green')
    ax.plot(depths, fidelity['mms'], label = 'MMS', linewidth = 4, color='orange')
    ax.fill_between(depths, fidelity['mms'] - fidelity['mms_std'], fidelity['mms'] + fidelity['mms_std'], alpha=0.2, color='orange')
    ax.set(xlabel='Circuit Depth', ylabel='Fidelity', xticks=depths)


else:
    fig=plt.figure(figsize=(12,7))
    ax=fig.add_subplot(111)

    ax.plot(depths, trace['model'], label = 'RL', linewidth=4, color='#e60049')
    ax.fill_between(depths, trace['model'] - trace['std_model'], trace['model'] + trace['std_model'], alpha=0.2, color='#e60049')
    ax.plot(depths, trace['RB'], label='RB', linewidth = 4, color='#0bb4ff')
    ax.fill_between(depths, trace['RB'] - trace['std_RB'], trace['RB'] + trace['std_RB'], alpha=0.2, color='#0bb4ff')
    ax.plot(depths, trace['no_noise'], label = 'No noise added', linewidth = 4, color='green')
    ax.fill_between(depths, trace['no_noise'] - trace['no_noise_std'], trace['no_noise'] + trace['no_noise_std'], alpha=0.2, color='green')
    ax.plot(depths, trace['mms'], label = 'MMS', linewidth = 4, color='orange')
    ax.fill_between(depths, trace['mms'] - trace['mms_std'], trace['mms'] + trace['mms_std'], alpha=0.2, color='orange')
    ax.set(xlabel='Circuit Depth', ylabel='Trace Distance', xticks=depths)
    ax.set_ylim([0, 1])
    ax.legend(ncol = 2)


plt.savefig(exp_folder + "images/RL_RB_comparison.pdf", )
plt.show()