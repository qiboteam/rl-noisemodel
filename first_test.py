from dataset import Dataset
import numpy as np
from acme_env import CircuitEnv

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

print('Circuit')
for c in dataset.get_circuits():
    print(c.draw())
circuit_repr=dataset.generate_dataset_representation()
print('Representation')
print(circuit_repr[0])
np.save('data/first_test_dataset.npy', circuit_repr)

circuit_env=CircuitEnv(circuit_repr[0])
print(circuit_env.observation_spec())
print(circuit_env.action_spec())