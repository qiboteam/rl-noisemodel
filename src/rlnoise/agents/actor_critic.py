import numpy as np
import tensorflow as tf
from tensorflow import keras
from rlnoise.envs.gym_env import CircuitsGym
from rlnoise.utils import models_folder

class AC_agent(object):

    def __init__(self, model, env, train_val_split=0.2):
        self.model=model
        self.env=env
        self.split=train_val_split

    def train(self, episodes, optimizer, verbose=True, save_model=True, filename="model_1q"):
        action_probs_history = []
        critic_value_history = []
        reward_history = []
        final_observation_history = []
        huber_loss = keras.losses.Huber()
        num_actions = self.env.action_space.n

        for episode in range(episodes):
            if ((episode+1)%10)==0 and verbose:
                print("episode: %d, reward %f" % (episode+1, reward))
            # initialize the environement
            state = self.env.reset() 
            done = False
            with tf.GradientTape() as tape:
                while not done:
                    state = tf.convert_to_tensor(state)
                    # add a dimension to the state (4) -> (1,4) as required by tf/keras
                    state = tf.expand_dims(state, 0)
                    # given the state predict the probability of each action and the future rewards 
                    action_probs, critic_value = self.model(state)
                    critic_value_history.append(critic_value[0, 0])
                    # compute a new action on the base of the predicted probabilities
                    action = np.random.choice(num_actions, p=np.squeeze(action_probs))
                    action_probs_history.append(tf.math.log(action_probs[0, action]))
                    # applies the new action
                    state, reward, done = self.env.step(action)

                # compute of the loss for the actor and critic 
                history = zip(action_probs_history, critic_value_history)
                actor_losses = []
                critic_losses = []
                for log_prob, value in history:
                    # for each value in the history the critic had estimated to have in the future a total reward of "value"
                    # we have taken a given action with log(prob) = log_prob and we have received the reward "ret"
                    # according to this we need to update the actor so that he will predict with higher probability an action 
                    # that will bring higher reward with respect to the one estimated by the critic 
                    diff = reward - value
                    actor_losses.append(-log_prob * diff)  # actor loss
                    # and we need to update the critic so that it will predict a better estimate of the total future rewards
                    critic_losses.append(
                        huber_loss(tf.expand_dims(value, 0), tf.expand_dims(reward, 0))
                    )
                # Backpropagation
                loss_value = sum(actor_losses) + sum(critic_losses)
                grads = tape.gradient(loss_value, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                # Clear the loss and reward history
                action_probs_history.clear()
                critic_value_history.clear()
            # saves details to monitor trainign and produce useful printout
            reward_history.append(reward)
            final_observation_history.append(state)
        if save_model:
            self.model.save(models_folder() + '/' + filename)
        return reward_history, final_observation_history

    def validation(self):
        '''Implement validation step'''
        return