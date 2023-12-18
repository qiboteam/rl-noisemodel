import os
import json
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from rlnoise.utils import model_evaluation

SAVE_TRAIN_DATA = True

config_path = f'{str(Path().parent.absolute())}/src/rlnoise/config.json'
with open(config_path) as f:
    config = json.load(f)

class CNNFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(
            self,
            observation_space,
            features_dim,
            filter_shape,
            n_filters = 64                 
    ):
        super().__init__(observation_space, features_dim)
        indim = observation_space.shape[0]
        sample = torch.as_tensor(observation_space.sample()[None]).float()
        conv1 = torch.nn.Conv2d( in_channels=indim,out_channels=n_filters, 
                                kernel_size=filter_shape) #adding pooling layer?
        
        # Compute shape by doing one forward pass
        # with torch.no_grad():
        #     shape = conv1(sample).shape
            
        self.cnn = torch.nn.Sequential(
            conv1,
            torch.nn.ReLU(), # Relu might not be great if we have negative angles, ELU
            torch.nn.Flatten(1,-1),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            hdim = self.cnn(sample).shape[-1]

        if hdim < features_dim:
            print(f'Warning, using features_dim ({features_dim}) greater than hidden dim ({hdim}).')

        self.linear = torch.nn.Sequential(torch.nn.Linear(hdim, features_dim), torch.nn.ELU())

    def forward(self, x):
        x = self.cnn(x)  
        return self.linear(x)
    

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    Args:
        check_freq: number of steps after wich will be performed the callback
        evaluation_set: np array loaded with np.load('bench_dataset') 
        train_environment: object of class QuantumCircuit()
        trainset_depth: number of gates per qubit used in the bench dataset

    """
    def __init__(self, check_freq, evaluation_set, train_environment, trainset_depth, verbose=False ,test_on_data_size = None, result_filename=None):
        super(CustomCallback, self).__init__(verbose)

        policy_params  =  config['policy']
        self.save_best = policy_params['save_best_model']
        self.plot = policy_params['plot_results']
        self.best_model_name = policy_params['model_name']
        self.plot_name = policy_params['plot_name']
        self.log_dir = f'{str(Path().parent.absolute())}/src/rlnoise/saved_models/'
        self.plot_dir = (
            f'{str(Path().parent.absolute())}/src/'
        )
        self.results_path = result_filename
        self.check_freq = check_freq
        self.test_on_data_size = test_on_data_size
        self.environment = train_environment
        self.best_mean_reward = -np.inf
        self.best_mean_fidelity = -np.inf
        self.best_mean_trace_dist = np.inf
        self.best_mean_bures_dist = np.inf
        if self.test_on_data_size is not None:
            self.train_circ = evaluation_set['train_set'][:self.test_on_data_size]
            self.train_label = evaluation_set['train_label'][:self.test_on_data_size]
        else:
            self.train_circ = evaluation_set['train_set']
            self.train_label = evaluation_set['train_label']

        self.val_circ = evaluation_set['val_set']
        self.val_label = evaluation_set['val_label']
        self.dataset_size = len(self.train_circ)
        self.trainset_depth = trainset_depth
        self.n_qubits = self.train_circ[0].shape[1]
        self.eval_results = []
        self.train_results = []
        self.timestep_list = []
        self.save_path = os.path.join(self.log_dir, self.best_model_name)
        self.plot_1_title = '%dQ D%d K3 logReward,Penal=0, Trainset_size=%d Valset_size=%d, p0=0.05 lam=0.05 e_z=0.1 e_x=0.05'%(self.n_qubits,self.trainset_depth,self.dataset_size,len(self.val_circ))
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

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
        if self.n_calls == 1 or self.n_calls % self.check_freq == 0:
          # Retrieve training reward
            training_results = model_evaluation(self.train_circ,self.train_label,model=self.model)
            evaluation_results = model_evaluation(self.val_circ,self.val_label,model=self.model)
            self.train_results.append(training_results)
            self.eval_results.append(evaluation_results)
            self.timestep_list.append(self.num_timesteps)

            if self.verbose:
                print(f"Num timesteps: {self.num_timesteps}")
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, evaluation_results['reward'].item()))
                print('Standard deviations: Reward_std={:.2f}, Fidelity_std={:.2f}, Trace distance_std={:.2f}'.format(
                    evaluation_results["reward_std"].item(),evaluation_results["fidelity_std"].item(),evaluation_results["trace_distance_std"].item()))
                        #print('Average correction applied: \n',evaluation_results["avg_correction"])
            if self.save_best is True:
                self.save_best_model(evaluation_results["reward"])
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
                self.results_path, 
                timesteps=self.timestep_list, 
                train_results=self.train_results, 
                val_results=self.eval_results,
                allow_pickle = True)
        
        if self.plot is True:
            self.plot_results()
        '''
        if self.test_on_data_size is not None:
            f = open(self.results_path+"test_size%d_D_%d_Dep-Term_CZ_3Q.npz"%(self.dataset_size,self.trainset_depth),"wb")
            np.savez(f,train_results=self.train_results, val_results=self.eval_results, timesteps=self.timestep_list)
            f.close()
        ''' 

    def plot_results(self):
        train_results = self.train_results.reshape(-1)
        eval_results = self.eval_results.reshape(-1)
        time_steps = self.timestep_list
        if self.test_on_data_size is None:
            self._extracted_from_plot_results_7(time_steps, eval_results, train_results)
        else:
            self._extracted_from_plot_results_28(time_steps, eval_results, train_results)

    # TODO Rename this here and in `plot_results`
    def _extracted_from_plot_results_28(self, time_steps, eval_results, train_results):
        fig=plt.figure(figsize=(5,5))
        fig.suptitle('1Q w CZ D5 K3 SR-off train dataset size=%d, val dataset size=%d'%(self.dataset_size,len(self.val_circ)) , fontsize=15)
        ax=fig.add_subplot(111)
        ax.set(xlabel='timesteps/1000', ylabel='Reward',title='Average final reward')
        ax.plot(time_steps,eval_results[:,0],label='evaluation_set',marker='x')
        ax.plot(time_steps,train_results[:,0],color='orange',label='train_set',marker='x')
        ax.legend()
        fig.savefig(self.plot_dir+'test_size%d_D_%d_Dep-Term_CZ_%dQ.png'%(self.dataset_size,self.trainset_depth,self.n_qubits),dpi=300)

    # TODO Rename this here and in `plot_results`
    def _extracted_from_plot_results_7(self, time_steps, eval_results, train_results):
        fig, ax = plt.subplots(2, 2, figsize=(15, 8))
        fig.suptitle(self.plot_1_title, fontsize=15)

        plt.subplots_adjust(left=0.168, bottom=0.06, right=0.865, top=0.92, wspace=0.207, hspace=0.21)
        errorevery=20
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
        fig.savefig(self.plot_dir+self.plot_name+'_Q%d_D%d_steps%d.png'%(self.n_qubits,self.trainset_depth,self.timestep_list[-1]),dpi=300)
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
    def save_best_model(self,reward):
        if reward.item() > self.best_mean_reward:
            if self.verbose:
                print(f"Saving new best model at {self.num_timesteps} timesteps")
                print(f"Saving new best model to {self.save_path}.zip")
            self.model.save(self.save_path+str(self.num_timesteps))
        