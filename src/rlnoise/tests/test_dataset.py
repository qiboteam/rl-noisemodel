from rlnoise.dataset import Dataset, dataset_folder, figures_folder
import numpy as np
import matplotlib.pyplot as plt

nqubits = 1
ngates = 2
ncirc = 1

# create dataset
dataset = Dataset(
    n_circuits=ncirc,
    n_gates=ngates,
    n_qubits=nqubits,
)
print('Circuits')
for c in dataset.get_circuits():
    print(c.draw())
    

print('-------------------------------------')
print('Noisy circuits')
dataset.add_noise(noisy_gates=['rx'])
for c in dataset.get_noisy_circuits():
    print(c.draw())

print('-------------------------------------')
print('Z observable')
z=dataset.pauli_probabilities(observable='Z', n_shots=100, n_rounds=1000)
plt.hist(z[0], density=True)

plt.savefig(figures_folder() + '/z_obs.png')
plt.close()


print('-------------------------------------')
print('Representation')
repr=dataset.generate_dataset_representation()
print(repr.shape)
np.save(dataset_folder() + '/test.npy', repr)

'''
print('-------------------------------------')
print('Y observable')
y=dataset.pauli_probabilities(observable='Y', n_shots=100, n_rounds=10)
plt.hist(y[0], density=True)
plt.savefig(figures_folder() + '/y_obs.png')
plt.close()

print('-------------------------------------')
print('X observable')
x=dataset.pauli_probabilities(observable='X', n_shots=100, n_rounds=10)
plt.hist(x[0], density=True)
plt.savefig(figures_folder()+ '/x_obs.png')
plt.close()

print('-------------------------------------')
print('Shots')
shots=dataset.noisy_shots(n_shots=2048, probabilities=True)
print(shots)
np.save('data/noisy_shots_1q.npy', repr)

dataset.train_val_split()
print('> Training Circuits')
for c in dataset.get_train_loader():
    print(c.draw())
print('> Validation Circuits')
for c in dataset.get_val_loader():
    print(c.draw())

print('Saving Circuits to dataset.json')
dataset.save_circuits('data/dataset1q.json')
dataset.load_circuits('data/dataset1q.json')
print('Loading Circuits from dataset.json\n')

for i in range(len(dataset)):
    print(dataset[i].draw())
'''