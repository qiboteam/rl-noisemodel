from rlnoise.dataset import Dataset
from rlnoise.agents.actor_critic_V2 import AC_agent
from tensorflow import keras
from rlnoise.envs.gym_env import CircuitsGym
from rlnoise.utils import truncated_moments_matching

nqubits = 1
ngates = 5
ncirc = 100

# create dataset
print("Creating random cicuits")
dataset = Dataset(
    n_circuits=ncirc,
    n_gates=ngates,
    n_qubits=nqubits,
)
print("Generating cicuits representation")
circuits_repr=dataset.generate_dataset_representation()
print("Adding noise")
dataset.add_noise(noise_params=0.1)
print("Measuring observables")
#labels=dataset.generate_labels()
labels=dataset.generate_dm_labels()

env=CircuitsGym(circuits_repr, labels, reward_func=truncated_moments_matching, reward_method='dm', reward_each_step=True)
#env.set_reward_func(truncated_moments_matching)
num_inputs = env.observation_space.shape
num_actions = env.action_space.n

# CNN model
inputs = keras.layers.Input(shape=num_inputs)
common = keras.layers.Conv1D(8, (1,), activation="relu")(inputs)
common = keras.layers.Conv1D(4, (3,), padding='same', activation="relu")(common)
common = keras.layers.Flatten()(common)
action = keras.layers.Dense(num_actions, activation="softmax", name='Actor')(common)
critic = keras.layers.Dense(1, name='Critic')(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])
model.summary()

agent=AC_agent(model, env)
agent.train_val_split(split=0.1)
agent.validation_options(do_validation=True, val_steps=100, greedy_policy=True)

optimizer = keras.optimizers.Adam(learning_rate=0.001)
episodes=1000

train_history, val_history = agent.train(episodes=episodes, optimizer=optimizer, verbose_episode=100)