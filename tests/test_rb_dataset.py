from rlnoise.randomized_benchmarking import rb_dataset_generator
import numpy as np

# config_file = 'hardware/experiments/qw11qB5/config_qibolab.json'

# rb_dataset_generator(config_file)

dataset = np.load("hardware/experiments/qw11qB5/rb_dataset.npz", allow_pickle=True)
circuits = dataset["circuits"]
print(circuits[1][1])