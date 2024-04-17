import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use('science')

exp_folder = "simulation/experiments/3q_high_noise/"
rb_result = exp_folder + "rb_result.npz"
rl_result = exp_folder + "rl_eval.npz"

fidelity = {}
with open(rb_result, "rb") as f:
    rb_result = np.load(f, allow_pickle=True)
    rb_result = rb_result['result']
    depths = rb_result['depth'].flatten()
    fidelity['RB'] = rb_result['fidelity'].flatten()
    fidelity['std_RB'] = rb_result['fidelity_std'].flatten()
    fidelity['no_noise'] = rb_result['fidelity_no_noise'].flatten()
    fidelity['no_noise_std'] = rb_result['fidelity_no_noise_std'].flatten()
with open(rl_result, "rb") as f:
    rl_result = np.load(f, allow_pickle=True)
    rl_result=rl_result['result']
    fidelity['model'] = rl_result['fidelity'].flatten()
    fidelity['std_model'] = rl_result['fidelity_std'].flatten()

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

fig=plt.figure(figsize=(12, 9))
ax=fig.add_subplot(111)
ax.plot(depths, fidelity['model'], label = 'RL-Model', linewidth=4, color='#e60049')
ax.fill_between(depths, fidelity['model'] - fidelity['std_model'], fidelity['model'] + fidelity['std_model'], alpha=0.2, color='#e60049')
ax.plot(depths, fidelity['RB'], label='RB', linewidth = 4, color='#0bb4ff')
ax.fill_between(depths, fidelity['RB'] - fidelity['std_RB'], fidelity['RB'] + fidelity['std_RB'], alpha=0.2, color='#0bb4ff')
ax.plot(depths, fidelity['no_noise'], label = 'No noise added', linewidth = 4, color='green')
ax.fill_between(depths, fidelity['no_noise'] - fidelity['no_noise_std'], fidelity['no_noise'] + fidelity['no_noise_std'], alpha=0.2, color='green')
ax.legend()
ax.set(xlabel='Circuit Depth', ylabel='Fidelity', xticks=depths)

plt.savefig("RL_RB_comparison.pdf", )
plt.show()