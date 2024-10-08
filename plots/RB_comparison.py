import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import scienceplots

plt.style.use('science')

logscale = False
exp_folder = "experiments/simulation/1q/"
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
    trace['RB'] = rb_result['trace'].flatten()
    trace['no_noise'] = rb_result['trace_no_noise'].flatten()
    trace['mms'] = rb_result['trace_mms'].flatten()
    trace['std_RB'] = rb_result['trace_std'].flatten()
    trace['no_noise_std'] = rb_result['trace_no_noise_std'].flatten()
    trace['mms_std'] = rb_result['trace_mms_std'].flatten()

with open(rl_result, "rb") as f:
    rl_result = np.load(f, allow_pickle=True)
    rl_result=rl_result['result']
    fidelity['model'] = rl_result['fidelity'].flatten()
    fidelity['std_model'] = rl_result['fidelity_std'].flatten()
    mse['model'] = rl_result['mse'].flatten()
    mse['std_model'] = rl_result['mse_std'].flatten()
    trace['model'] = rl_result['trace'].flatten()
    trace['std_model'] = rl_result['trace_std'].flatten()

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

fig, ax = plt.subplots(1, 2, figsize=(24, 9))

# Fidelity
ax[0].plot(depths, fidelity['model'], label = 'RL', linewidth=4, color='#e60049')
ax[0].fill_between(depths, fidelity['model'] - fidelity['std_model'], fidelity['model'] + fidelity['std_model'], alpha=0.2, color='#e60049')
ax[0].plot(depths, fidelity['RB'], label='RB', linewidth = 4, color='#0bb4ff')
ax[0].fill_between(depths, fidelity['RB'] - fidelity['std_RB'], fidelity['RB'] + fidelity['std_RB'], alpha=0.2, color='#0bb4ff')
ax[0].plot(depths, fidelity['no_noise'], label = 'No noise added', linewidth = 4, color='green')
ax[0].fill_between(depths, fidelity['no_noise'] - fidelity['no_noise_std'], fidelity['no_noise'] + fidelity['no_noise_std'], alpha=0.2, color='green')
ax[0].plot(depths, fidelity['mms'], label = 'MMS', linewidth = 4, color='orange')
ax[0].fill_between(depths, fidelity['mms'] - fidelity['mms_std'], fidelity['mms'] + fidelity['mms_std'], alpha=0.2, color='orange')
ax[0].legend()
ax[0].set(xlabel='Circuit Depth', ylabel='Fidelity', xticks=depths)

# Trace distance
if logscale:
    ax[1].set_yscale('log')
    # Use more ticks for log scale
    ax[1].yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs='auto'))
    ax[1].yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(1, 10)))
    ax[1].yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax[1].yaxis.set_minor_formatter(ticker.ScalarFormatter())
ax[1].plot(depths, trace['model'], label='RL-Model', linewidth=4, color='#e60049')
ax[1].fill_between(depths, trace['model'] - trace['std_model'], trace['model'] + trace['std_model'], alpha=0.2, color='#e60049')
ax[1].plot(depths, trace['RB'], label='RB', linewidth=4, color='#0bb4ff')
ax[1].fill_between(depths, trace['RB'] - trace['std_RB'], trace['RB'] + trace['std_RB'], alpha=0.2, color='#0bb4ff')
ax[1].plot(depths, trace['no_noise'], label='No noise added', linewidth=4, color='green')
ax[1].fill_between(depths, trace['no_noise'] - trace['no_noise_std'], trace['no_noise'] + trace['no_noise_std'], alpha=0.2, color='green')
ax[1].plot(depths, trace['mms'], label='MMS', linewidth=4, color='orange')
ax[1].fill_between(depths, trace['mms'] - trace['mms_std'], trace['mms'] + trace['mms_std'], alpha=0.2, color='orange')
ax[1].set(xlabel='Circuit Depth', ylabel='Trace Distance', xticks=depths)

plt.savefig(exp_folder + "images/RL_RB_comparison.pdf", )