from rlnoise.randomized_benchmarking import rb_dataset_generator
import numpy as np

config_file = 'tests/config_test.json'

rb_dataset_generator(config_file)

dataset = np.load("tests/rb_dataset.npz", allow_pickle=True)
circuits = dataset["circuits"]
print(circuits[5][0])