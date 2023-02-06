from rlnoise.dataset import Dataset
from rlnoise.envs.gym_env import CircuitsGym

episodes = 5
results = []  

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
labels=dataset.generate_labels()
print(labels)

circuits_env=CircuitsGym(circuits_repr, labels)
for episode in range(episodes):
    state = circuits_env.reset()
    done = False 
    total = 0
    while not done:
        action = circuits_env.action_space.sample()
        state, reward, done = circuits_env.step(action) 
        total += reward
    circuits_env._get_info(last_step=True)
    print("Reward: ", reward) 
    results.append(total)
    circuits_env.close()