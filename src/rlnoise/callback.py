import json
import numpy as np
from rlnoise.utils import model_evaluation
from stable_baselines3.common.callbacks import BaseCallback
from rlnoise.dataset import load_dataset

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``. This callback will be used observe the training and evaluation results.

    """
    def __init__(self, config_path: str):
        super(CustomCallback, self).__init__()
        
        with open(config_path, 'r') as f:
            config = json.load(f)

        callback_params  =  config['callback']
        self.save_best = callback_params['save_best_model']
        self.plot = callback_params['plot_results']
        model_name = callback_params['model_name']
        model_folder = callback_params['model_folder']
        results_folder = callback_params['results_folder']
        self.results_name = f"{results_folder}/{model_name}"
        self.save_path = f"{model_folder}/{model_name}"
        self.check_freq = callback_params['check_freq']
        self.verbose = callback_params['verbose']
        self.best_mean_reward = -np.inf
        self.best_mean_fidelity = -np.inf
        self.best_mean_trace_dist = np.inf

        self.circuits, self.labels, self.val_circuits, self.val_labels = load_dataset(self.dataset_file)
        
        self.dataset_size = len(self.train_circ)
        self.n_qubits = self.train_circ[0].shape[1]
        self.eval_results = []
        self.train_results = []
        self.timestep_list = []
        self.plot_1_title = "Training and evaluation results"


    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.n_calls==1 or self.n_calls % self.check_freq == 0:
          # Retrieve training reward
            training_results = model_evaluation(self.train_circ,self.train_label,model=self.model)
            evaluation_results = model_evaluation(self.val_circ,self.val_label,model=self.model)
            self.train_results.append(training_results)
            self.eval_results.append(evaluation_results)
            self.timestep_list.append(self.num_timesteps)

            if self.verbose:
                print(f"Num timesteps: {self.num_timesteps}")
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, evaluation_results['reward'].item()))
                print('Fidelity={:.2f}, Fidelity_std={:.2f}'.format(
                    evaluation_results["fidelity"].item(),evaluation_results["fidelity_std"].item()))
                        #print('Average correction applied: \n',evaluation_results["avg_correction"])
            if self.save_best is True:
                self.save_best_model(evaluation_results["fidelity"])
            self.save_best_results(evaluation_results["reward"],evaluation_results["fidelity"],evaluation_results["trace_distance"],evaluation_results["bures_distance"])

            debug = True

            if debug:
                print('Considering action: \n',self.environment._get_info()['Action'],' at last position')
                print('State AFTER action: \n',self.environment._get_info()['State_after'])
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
        print('Best average results obtained on the evaluation set are:\n Reward=%f, Fidelity=%f, Trace distance=%f, Bures=%f'%(self.best_mean_reward,self.best_mean_fidelity,self.best_mean_trace_dist,self.best_mean_bures_dist))
        
        if SAVE_TRAIN_DATA:
            np.savez(
                self.results_name, 
                timesteps=self.timestep_list, 
                train_results=self.train_results, 
                val_results=self.eval_results,
                allow_pickle = True)
        
        if self.plot is True:
            self.plot_results()
        '''
        if self.test_on_data_size is not None:
            f = open(self.results_path+"test_size%d_Dep-Term_CZ_3Q.npz"%(self.dataset_size),"wb")
            np.savez(f,train_results=self.train_results, val_results=self.eval_results, timesteps=self.timestep_list)
            f.close()
        ''' 

    def plot_results(self):
        train_results = self.train_results.reshape(-1)
        eval_results = self.eval_results.reshape(-1)
        time_steps = self.timestep_list
        if self.test_on_data_size is None:
            self.standard_plot(time_steps, eval_results, train_results)
        else:
            self.plot_specific_datasize(time_steps, eval_results, train_results)

    # TODO Rename this here and in `plot_results`
    def plot_specific_datasize(self, time_steps, eval_results, train_results):
        fig=plt.figure(figsize=(5,5))
        fig.suptitle('1Q w CZ D5 K3 SR-off train dataset size=%d, val dataset size=%d'%(self.dataset_size,len(self.val_circ)) , fontsize=15)
        ax=fig.add_subplot(111)
        ax.set(xlabel='timesteps/1000', ylabel='Reward',title='Average final reward')
        ax.plot(time_steps,eval_results[:,0],label='evaluation_set',marker='x')
        ax.plot(time_steps,train_results[:,0],color='orange',label='train_set',marker='x')
        ax.legend()
        fig.savefig(f"{self.save_path}/test_size{self.dataset_size}_Dep-Term_CZ_{self.n_qubits}Q.png",dpi=300)

    # TODO Rename this here and in `plot_results`
    def standard_plot(self, time_steps, eval_results, train_results):
        fig, ax = plt.subplots(2, 2, figsize=(15, 8))
        fig.suptitle(self.plot_1_title, fontsize=15)

        plt.subplots_adjust(left=0.168, bottom=0.06, right=0.865, top=0.92, wspace=0.207, hspace=0.21)
        errorevery=1
        ax[0,0].errorbar(time_steps,eval_results["reward"],yerr=0,label='evaluation_set',errorevery=errorevery,capsize=4, marker='.') #use list comprehension
        ax[0,0].set(xlabel='timesteps/1000', ylabel='Reward',title='Average final reward')
        ax[0,1].errorbar(time_steps,eval_results["fidelity"],yerr=eval_results["fidelity_std"],errorevery=errorevery,capsize=4, marker='.')
        ax[0,1].set(xlabel='timesteps/1000', ylabel='Fidelity',title='Fidelity between DM')
        ax[1,0].errorbar(time_steps,eval_results["trace_distance"],yerr=eval_results["trace_distance_std"],errorevery=errorevery,capsize=4, marker='.')
        ax[1,0].set(xlabel='timesteps/1000', ylabel='Trace Distance',title='Trace distance between DM')
        ax[1,1].errorbar(time_steps,eval_results["bures_distance"],yerr=eval_results["bures_distance_std"],errorevery=errorevery,capsize=4, marker='.')
        ax[1,1].set(xlabel='timesteps/1000', ylabel='Bures Distance',title='Bures distance between DM')

        ax[0,0].errorbar(time_steps,train_results["reward"],yerr=train_results["reward_std"],color='orange',label='train_set',errorevery=errorevery,capsize=4, marker='.')
        ax[0,1].errorbar(time_steps,train_results["fidelity"],yerr=train_results["fidelity_std"],color='orange',label='train_set',errorevery=errorevery,capsize=4, marker='.')
        ax[1,0].errorbar(time_steps,train_results["trace_distance"],yerr=train_results["trace_distance_std"],color='orange',label='train_set',errorevery=errorevery,capsize=4, marker='.')
        ax[1,1].errorbar(time_steps,train_results["bures_distance"],yerr=train_results["bures_distance_std"],color='orange',label='train_set',errorevery=errorevery,capsize=4, marker='.')
        ax[0,0].legend()
        fig.savefig(f"{self.save_path}/{self.plot_name}_steps{self.timestep_list[-1]}.png",dpi=300)
        plt.show()
            #plt.show()
        

    def generalization_test():
        #here will be tested the simple generalization (1 depth for train and different for test)
        # and a more complex one (same depth but test circuit sligthly different noise parameter)
        pass

    def save_best_results(self,reward,fidelity,trace_dist,bures_dist): 
        reward = reward.item() 
        fidelity = fidelity.item()
        trace_dist = trace_dist.item()
        if fidelity > self.best_mean_fidelity:
            self.best_mean_reward = reward
            self.best_mean_fidelity = fidelity
            self.best_mean_trace_dist = trace_dist           
            self.best_mean_bures_dist = bures_dist

    def save_best_model(self,fidelity):
        if fidelity.item() >= self.best_mean_fidelity:
            if self.verbose:
                print(f"Saving new best model at {self.num_timesteps} timesteps")
                print(f"Saving new best model in {self.save_path}")
            self.model.save(f"{self.save_path}/{self.best_model_name}_{str(self.num_timesteps)}.zip")
        