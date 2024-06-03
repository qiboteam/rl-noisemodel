import matplotlib.pyplot as plt
import numpy as np
import scienceplots
plt.style.use('science')


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

# Load the data
data = np.load("simulation/experiments/test_size/size_test_result.npz", allow_pickle=True)

sizes = data['result']['size'].flatten()
fidelity_means = data['result']['fidelity_mean'].flatten()
mse_means = data['result']['mse_mean'].flatten()
trace_distance_means = data['result']['trace_distance_mean'].flatten()
fidelity_stds = data['result']['fidelity_std'].flatten()
mse_stds = data['result']['mse_std'].flatten()
trace_distance_stds = data['result']['trace_distance_std'].flatten()

print(sizes)

plt.figure(figsize=(10, 6))
plt.errorbar(sizes, trace_distance_means, yerr=trace_distance_stds, capsize=5)
plt.xlabel('Size')
plt.ylabel('Trace Distance')

# Show the plot
plt.show()