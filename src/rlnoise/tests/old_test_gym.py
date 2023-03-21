from rlnoise.old_dataset import Dataset
import numpy as np
from rlnoise.envs.old_gym import CircuitsGym

nqubits = 1
ngates = 5
ncirc = 2
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
labels=dataset.generate_dm_labels()
print(labels)


circuit_env=CircuitsGym(circuits_repr, labels, reward_method="dm")
circuit_env.reset(verbose=True)
for _ in range(ngates):
    action=np.random.randint(0,2)
    circuit_env.step(action, verbose=True)
