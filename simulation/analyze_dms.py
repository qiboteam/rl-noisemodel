import numpy as np
import matplotlib.pyplot as plt
from rlnoise.utils import trace_distance

exp_folder = "simulation/experiments/3q_low_noise/"
ground_truth = exp_folder + "eval_dataset.npz"
noisy_dm = exp_folder + "evaluation_result.npz"
metric = np.abs

with open(ground_truth, "rb") as f:
    dataset = np.load(f, allow_pickle=True)
    gt_labels = dataset['labels']

with np.load(noisy_dm, allow_pickle=True) as data:
    result_dm = data['dms']

print("Len Dataset: ", gt_labels.shape)
assert gt_labels.shape == result_dm.shape
trace_distances = np.array([trace_distance(result_dm[i], gt_labels[i]) for i in range(len(gt_labels))], dtype=float)
average_labels = np.mean(np.array([metric(result_dm[i], gt_labels[i]) for i in range(len(gt_labels))], dtype=float), axis=0)
print("Average error: ", np.mean(average_labels))
print("Average diagonal error: ", np.mean(average_labels.diagonal()))
print("Average trace distance: ", np.mean(trace_distances))
#print("DM error:\n", average_labels)

# Create a heatmap
plt.imshow(average_labels, cmap='hot')
plt.colorbar(label='Average error')
plt.show()



