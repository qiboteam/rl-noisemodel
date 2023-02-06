from dataset import Dataset
import numpy as np
from acme_env import CircuitsEnv
import matplotlib.pyplot as plt

nqubits = 1
ngates = 5
ncirc = 1
val_split=0.2

# create dataset
dataset = Dataset(
    n_circuits=ncirc,
    n_gates=ngates,
    n_qubits=nqubits,
)

print('Circuits')
for c in dataset.get_circuits():
    print(c.draw())
circuits_repr=dataset.generate_dataset_representation()
dataset.add_noise(noise_params=0.05)
labels=dataset.generate_labels()
print(labels)


circuit_env=CircuitsEnv(circuits_repr, labels)
circuit_env.reset()
for _ in range(ngates):
    action=np.random.randint(0,2)
    circuit_env.step(action)
    circuit_env.get_info()

reward=circuit_env.compute_reward(labels[0])
print(reward)