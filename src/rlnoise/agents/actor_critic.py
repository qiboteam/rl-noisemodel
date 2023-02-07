from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow import keras

from rlnoise.dataset import Dataset
from rlnoise.envs.gym_env import CircuitsGym
from rlnoise.utils import plot_results
  
# create dataset
nqubits = 1
ngates = 5
ncirc = 2
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

env=CircuitsGym(circuits_repr, labels)

# AC agent
episodes = 5
results = []
num_inputs = env.observation_space.shape
num_actions = env.action_space.n
num_hidden = 128

# CNN model
inputs = keras.layers.Input(shape=num_inputs)
common = keras.layers.Conv1D(8, (1,), activation="relu")(inputs)
common = keras.layers.Conv1D(4, (3,), padding='same', activation="relu")(common)
common = keras.layers.Flatten()(common)
action = keras.layers.Dense(num_actions, activation="softmax", name='Actor')(common)
critic = keras.layers.Dense(1, name='Critic')(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])
model.summary()

optimizer = keras.optimizers.Adam(learning_rate=0.01)
# Loss with an inermediate behaviour between MAE and MSE (MSE around zero, MAE at lager values)
huber_loss = keras.losses.Huber() 
eps = np.finfo(np.float32).eps.item() 

# useful containers
action_probs_history = []
critic_value_history = []
reward = 0.
running_reward = 0
episode_count = 0
results = []
results_rr = []

for _ in range(episodes): 
    # initialize the environement
    state = env.reset() 
    episode_reward = 0.
    with tf.GradientTape() as tape:
        while not done:
            state = tf.convert_to_tensor(state)
            # add a dimension to the state (4) -> (1,4) as required by tf/keras
            state = tf.expand_dims(state, 0)
            # given the state predict the probability of each action and the future rewards 
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])
            # compute a new action on the base of the predicted probabilities
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))
            # applies the new action
            state, reward, done = env.step(action)

        # computes the epected value for the reward 
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for the critic

        # normalization
        returns = np.array(rewards_history)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # compute of the loss for the actor and critic 
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
    
            # for each value in the history the critic had estimated to have in the future a total reward of "value"
            # we have taken a given action with log(prob) = log_prob and we have received the reward "ret"

            # according to this we need to update the actor so that he will predict with higher probability an action 
            # that will bring higher rewqard with respect to the one estimated by the critic 
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # and we need to update the critic so that it will predict a better estimate of the total future rewards
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # saves details to monitor trainign and produce useful printout
    episode_count += 1
    results_rr.append(running_reward)
    results.append(episode_reward)
    plot_results(results,title='GP Actor-Critic Strategy')
    
    if running_reward > 195:  # Condition for which the task is considered solved
        print("Solved at episode {}!".format(episode_count))
        break



#printout of resuts
for episodio in range(episode_count):
    if episodio % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(results_rr[episodio], episodio))

    if results_rr[episodio] > 195:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episodio))
        break

for episode in range(episodes):
    state = env.reset()
    done = False 
    total = 0
    while not done:
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)
        action_probs, critic_value = model(state)
        # compute a new action on the base of the predicted probabilities
        action = np.random.choice(num_actions, p=np.squeeze(action_probs))
        state, reward, done = env.step(action) 
        total += reward
    env._get_info(last_step=True)
    print("Reward: ", reward) 
    results.append(total)
    env.close()