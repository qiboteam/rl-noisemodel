import json
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from rlnoise.utils import compute_fidelity, trace_distance

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``. This callback will be used observe the training and evaluation results.

    """
    def __init__(self, config_path: str, env):
        super(CustomCallback, self).__init__()
        self.env = env
        
        with open(config_path, 'r') as f:
            config = json.load(f)

        callback_params  =  config['callback']
        self.save_best = callback_params['save_best_model']
        self.plot = callback_params['plot_results']
        model_name = callback_params['model_name']
        results_folder = callback_params['result_folder']
        self.results_path = f"{results_folder}/{model_name}"
        self.save_path = f"{results_folder}/{model_name}"
        self.check_freq = callback_params['check_freq']
        self.verbose = callback_params['verbose']
        self.best_mean_fidelity = -np.inf

        self.eval_results = []
        self.train_results = []
        self.timestep_list = []

    def model_evaluation(self, train_set):
        '''
        Function for evaluating the model
        Args:
            train_set: bool, whether to evaluate on training or validation set.
        Returns:
            avg reward, avg fidelity, avg trace distance
        '''
        avg_rew = []
        avg_trace_distance = []
        avg_fidelity = []

        if train_set:
            start = 0
            stop = self.env.n_circ_train
        else:
            start = self.env.n_circ_train
            stop = self.env.n_circ

        for i in range(start, stop):
            obs, _ = self.env.reset(i=i)
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)       
                obs, reward, done, truncated, info = self.env.step(action)
            predicted_circ = self.env.get_qibo_circuit()
            dm_model = predicted_circ().state()

            avg_rew.append(reward)
            avg_fidelity.append(compute_fidelity(self.env.labels[i], dm_model))
            avg_trace_distance.append(trace_distance(self.env.labels[i], dm_model))

        rew = np.array(avg_rew)
        fid = np.array(avg_fidelity)
        trace_d = np.array(avg_trace_distance)

        return  np.array([(
                    rew.mean(),
                    rew.std(),
                    fid.mean(),
                    fid.std(),
                    trace_d.mean(),
                    trace_d.std()
                )],
                dtype = [
                    ("reward", '<f4'),
                    ("reward_std", '<f4'),
                    ("fidelity", '<f4'),
                    ("fidelity_std", '<f4'),
                    ("trace_distance", '<f4'),
                    ("trace_distance_std", '<f4')
                ])

                


    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.n_calls==1 or self.n_calls % self.check_freq == 0:
            training_results = self.model_evaluation(train_set = True)
            evaluation_results = self.model_evaluation(train_set = False)
            self.train_results.append(training_results)
            self.eval_results.append(evaluation_results)
            self.timestep_list.append(self.num_timesteps)

            if self.verbose:
                print(f"Timesteps: {self.num_timesteps}")
                print("Reward: ", training_results["reward"].item())
                print('Test set Fidelity: {:.2f} std: {:.2f}'.format(evaluation_results["fidelity"].item(), evaluation_results["fidelity_std"].item()))
            if self.save_best is True:
                if evaluation_results["fidelity"] >= self.best_mean_fidelity:
                    if self.verbose:
                        print(f"Saving new best model at {self.num_timesteps} timesteps.")
                        print(f"Saving new best model in {self.save_path}.")
                    self.model.save(f"{self.save_path}.zip")
                    self.best_mean_fidelity = evaluation_results["fidelity"]
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self.train_results = np.asarray(self.train_results)
        self.eval_results = np.asarray(self.eval_results)
        self.timestep_list = np.asarray(self.timestep_list)/1000

        np.savez(
            self.results_path, 
            timesteps = self.timestep_list, 
            train_results = self.train_results, 
            val_results = self.eval_results,
            allow_pickle = True)
        
        if self.plot is True:
            self.plot_results()

    def plot_results(self):
        import matplotlib.pyplot as plt

        train_results = self.train_results.reshape(-1)
        eval_results = self.eval_results.reshape(-1)
        time_steps = self.timestep_list

        fig, ax = plt.subplots(1, 3, figsize=(21, 7))
        errorevery = 1

        ax[0].errorbar(time_steps, eval_results["reward"], yerr=0,label='evaluation_set',errorevery=errorevery,capsize=4, marker='.')
        ax[0].set(xlabel='timesteps/1000', ylabel='Reward',title='Reward')
        ax[1].errorbar(time_steps,eval_results["fidelity"],yerr=eval_results["fidelity_std"],errorevery=errorevery,capsize=4, marker='.')
        ax[1].set(xlabel='timesteps/1000', ylabel='Fidelity',title='Fidelity')
        ax[2].errorbar(time_steps,eval_results["trace_distance"],yerr=eval_results["trace_distance_std"],errorevery=errorevery,capsize=4, marker='.')
        ax[2].set(xlabel='timesteps/1000', ylabel='Trace Distance',title='Trace distance')

        ax[0].errorbar(time_steps,train_results["reward"],yerr=train_results["reward_std"],color='orange',label='train_set',errorevery=errorevery,capsize=4, marker='.')
        ax[1].errorbar(time_steps,train_results["fidelity"],yerr=train_results["fidelity_std"],color='orange',label='train_set',errorevery=errorevery,capsize=4, marker='.')
        ax[2].errorbar(time_steps,train_results["trace_distance"],yerr=train_results["trace_distance_std"],color='orange',label='train_set',errorevery=errorevery,capsize=4, marker='.')
        ax[0].legend()
        plt.show()         

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass