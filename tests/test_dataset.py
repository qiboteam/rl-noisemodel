from rlnoise.dataset import Dataset, load_dataset
import numpy as np

config_file = "tests/config_test.json"
save_path = "tests/test_dataset.npz"

dataset = Dataset(config_file)

print("Circuits: ", dataset.n_circuits)
assert dataset.n_circuits == len(dataset.circ_rep), "Error in number of generated circuits"
print("Qubits: ", dataset.n_qubits)
assert dataset.n_qubits == dataset.circ_rep[0].shape[1], "Error in number of qubits"

print("First non-noisy circuit:")
print(dataset.circuits[0].draw())
print("First noisy circuit:")
print(dataset.noisy_circuits[0].draw())
print("First circuit representation:")
print(dataset.circ_rep[0])

dataset.save(save_path)
loaded_dataset, labels = load_dataset(save_path)

print("First loaded representation:")
print(loaded_dataset[0])
assert np.allclose(loaded_dataset[0], dataset.circ_rep[0]), "Error in loaded dataset"
print("First loaded label:")
print(labels[0])
assert np.allclose(labels[0], dataset.dm_labels[0]), "Error in loaded labels"

