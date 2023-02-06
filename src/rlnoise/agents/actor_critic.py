import os
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import tensorflow as tf
from tensorflow import keras
import gym

from rlnoise.dataset import Dataset
import numpy as np
from rlnoise.envs.gym_env import CircuitsGym

def plot_results(values, title=''):   
    # Update the window after each episode
    display.clear_output(wait=True)
    
    # Define the figure
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    f.suptitle(title)
    ax[0].plot(values, label='score per run')
    ax[0].axhline(195, c='red',ls='--', label='goal')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    #ax[0].set_ylim([0,200]) 
    x = range(len(values))
    ax[0].legend()
    # Calculate the trend
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0].plot(x,p(x),"--", label='trend')
    except:
        print('')
    
    # Plot the histogram of results
    ax[1].hist(values[-50:])
    ax[1].axvline(195, c='red', label='goal')
    ax[1].set_xlabel('Scores per Last 50 Episodes')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    plt.show()

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

env=CircuitsGym(circuits_repr, labels)

num_inputs = env.observation_space.shape
num_actions = env.action_space.n
num_hidden = 128

inputs = keras.layers.Input(shape=num_inputs)
common = keras.layers.Conv1D(8, (1,), activation="relu")(inputs)
common = keras.layers.Conv1D(4, (3,), padding='same', activation="relu")(common)
common = keras.layers.Flatten()(common)
action = keras.layers.Dense(num_actions, activation="softmax", name='Actor')(common)
critic = keras.layers.Dense(1, name='Critic')(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])
model.summary()

optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber() #a loss with an inermediate behaviour between MAE and MSE (MSE around zero, MAE at lager values)
# Parameters
gamma = 1.  # Discount factor for past rewards
max_steps_per_episode = 100
eps = np.finfo(np.float32).eps.item() # return the smallest folat so that: 1.0 + eps != 1.0

# useful containers
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0
results = []
results_rr = []

while True:  # Run until solved
    state = env.reset()  #initialize the environement
    episode_reward = 0

    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0) #add a dimension to the state (4) -> (1,4) as required by tf/keras
            # given the state predict the probability of each action and the future rewards 
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])
            # compute a new action on the base of the predicted probabilities
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))
            # applies the new action
            state, reward, done = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward

            if done:
                break

        # update the  running reward 
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        # computes the epected value for the reward 
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for the critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # normalization
        returns = np.array(returns)
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