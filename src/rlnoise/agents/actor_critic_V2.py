import numpy as np
import tensorflow as tf
from tensorflow import keras
from rlnoise.envs.gym_env import CircuitsGym
from rlnoise.utils import models_folder
import math

class AC_agent(object):

    def __init__(self, model, env, train_val_split=0.2):
        self.model=model
        self.env=env
        self.split=train_val_split
        self.val=False

    def validation_set(self, val_env):
        '''Load validation environment'''
        self.val_env=val_env

    def train_val_split(self, split=0.2):
        '''Split environment into train and validation sets'''
        tot=self.env.n_elements()
        circuits_repr=self.env.get_circuits_repr()
        labels=self.env.get_labels()
        n_split=math.floor((1-split)*tot)
        cr_train=circuits_repr[0:n_split]
        labels_train=labels[0:n_split]
        cr_val=circuits_repr[n_split:]
        labels_val=labels[n_split:]
        reward_func=self.env.get_reward_func()
        reward_method=self.env.get_reward_method()
        self.env=CircuitsGym(cr_train,labels_train, reward_func=reward_func, reward_method=reward_method)
        self.val_env=CircuitsGym(cr_val,labels_val, reward_func=reward_func, reward_method=reward_method)
        print("Train set elements: ", n_split)
        print("Validation set elements: ", tot-n_split)
        

    def validation_options(self, do_validation=True, val_steps=10, greedy_policy=True):
        '''Define options for validation
        do_validation (bool): define if to do or not the validation step
        val_steps (int): define after how many steps to do validation
        geedy_policy (bool): apply a greedy policy during validation
        '''
        if not self.val_env:
            print("Fist define validation set!")
        else:
            self.val_steps=val_steps
            self.do_val=do_validation
            self.greedy_val=greedy_policy
        
    def train_episode(self, optimizer):
        '''Run a training episode'''
        huber_loss = keras.losses.Huber()
        # Discount factor for past rewards
        gamma = 0.95  
        # Return the smallest folat so that: 1.0 + eps != 1.0
        eps = np.finfo(np.float32).eps.item() 
        # Useful containers
        action_probs_history = []
        critic_value_history = []
        rewards_history = []
        # Initialize the environement
        state = self.env.reset() 
        num_actions = self.env.action_space.n
        circuit = self.env.get_sample() 
        episode_reward = 0.
        done = False
        with tf.GradientTape() as tape:
            while done is not True:
                state = tf.convert_to_tensor(state)
                state = tf.expand_dims(state, 0)
                # given the state predict the probability of each action and the future rewards 
                action_probs, critic_value = self.model(state)
                critic_value_history.append(critic_value[0, 0])
                # compute a new action on the base of the predicted probabilities
                action = np.random.choice(num_actions, p=np.squeeze(action_probs))
                action_probs_history.append(tf.math.log(action_probs[0, action]))
                # applies the new action
                state, reward, done = self.env.step(action)
                rewards_history.append(reward)
                episode_reward += reward
            
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
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            # Clear the loss and reward history
            action_probs_history.clear()
            critic_value_history.clear()
            rewards_history.clear()
        return episode_reward, state, circuit

    def train(self, episodes, optimizer, verbose=True, verbose_episode=20):
        '''Implement full training pipeline'''
        train_history = []
        val_history=[]
        for episode in range(episodes):
            step_dictionary={}
            reward, state, circuit = self.train_episode(optimizer)
            step_dictionary["circuit"]=circuit
            step_dictionary["reward"]=reward
            step_dictionary["final_state"]=state
            train_history.append(step_dictionary)
            if ((episode+1)%verbose_episode)==0 and verbose:
                avg_reward=0
                for i in range((episode-10), episode):
                    avg_reward+=train_history[i]["reward"]
                print("episode: %d, avg_reward %f" % (episode+1, avg_reward/10.))
            if self.do_val and ((episode+1)%self.val_steps)==0:
                print("Validation...")
                val_reward=self.validation_step()
                val_history.append(val_reward)
                print("episode: %d, val_reward %f" % (episode+1, val_reward))
        if self.do_val:
            return train_history, val_history
        else:
            return train_history

    def save_model(self, filename="model_1q"):
        '''Save Keras model'''
        self.model.save(models_folder() + '/' + filename)

    def validation_step(self):
        '''Implement validation step'''
        tot_val_reward=0.
        num_actions = self.val_env.action_space.n
        for val_circuit in range(self.val_env.n_elements()):
            state = self.val_env.reset(sample=val_circuit)
            done = False
            while not done:
                state = tf.convert_to_tensor(state)
                state = tf.expand_dims(state, 0)
                action_probs, _ = self.model(state)
                if self.greedy_val:
                    action = np.where(action_probs.numpy()==np.max(action_probs.numpy()))
                else:
                    action = np.random.choice(num_actions, p=np.squeeze(action_probs))
                state, reward, done = self.val_env.step(action)
            tot_val_reward+=reward
        avg_val_reward=tot_val_reward/self.val_env.n_elements()
        return avg_val_reward