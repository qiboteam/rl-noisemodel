import os
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from rlnoise.utils import model_evaluation
import matplotlib.pyplot as plt
class CNNFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(
            self,
            observation_space,
            features_dim,
            filter_shape,
            n_filters = 32                 
    ):
        super().__init__(observation_space, features_dim)
        indim = observation_space.shape[0]
        print('observation spac shape: ',observation_space.shape)
        sample = torch.as_tensor(observation_space.sample()[None]).float()
        #print('indim: ',indim)
        #print('filter shape: ',filter_shape)
        filter_shape=(1,2)
        conv1 = torch.nn.Conv2d( in_channels=indim,out_channels=64, kernel_size=filter_shape) #adding pooling layer?
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            #shape = torch.nn.functional.avg_pool2d(conv1(sample),kernel_size=filter_shape).shape
            shape = conv1(sample).shape
            
        #conv2 = torch.nn.Conv2d(n_filters, n_filters, (max(int(shape[2]/2), 1), shape[3]))  #aggiungi linear layer prima di conv1
        conv2 = torch.nn.Conv2d(64, 32, (max(int(shape[2]/2), 1), shape[3]))

        self.cnn = torch.nn.Sequential(
            conv1,
            torch.nn.ReLU(), # Relu might not be great if we have negative angles, ELU
            #torch.nn.AvgPool2d(kernel_size=filter_shape),
            conv2,
            torch.nn.ReLU(),
            torch.nn.Flatten(1,-1),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            hdim = self.cnn(sample).shape[-1]

        if hdim < features_dim:
            print(f'Warning, using features_dim ({features_dim}) greater than hidden dim ({hdim}).')

        self.linear = torch.nn.Sequential(torch.nn.Linear(hdim, features_dim), torch.nn.ELU())

    def forward(self, x):
        #test=np.array(torch.Tensor.cpu(x[0]))
        #print('input of feature extractor: \n',test.transpose(2,1,0))
        x = self.cnn(x)
        
        return self.linear(x)
    

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, check_freq,  evaluation_set,train_environment,trainset_depth, verbose=1,save_best=True,plot=True):
        super(CustomCallback, self).__init__(verbose)
        self.save_best=save_best
        self.plot=plot
        self.check_freq = check_freq
        self.log_dir = os.getcwd()+'/src/rlnoise/saved_models/'
        self.plot_dir=os.getcwd()+'/src/rlnoise/data_analysis/plots/'
        
        self.environment=train_environment
        self.best_mean_reward = -np.inf
        self.train_circ=evaluation_set['train_set']
        
        self.train_label=evaluation_set['train_label']
        self.val_circ=evaluation_set['val_set']
        self.val_label=evaluation_set['val_label']
        self.trainset_depth=trainset_depth
        self.n_qubits=self.train_circ[0].shape[1]
        self.eval_results=[]
        self.train_results=[]
        self.timestep_list=[]
        self.save_path = os.path.join(self.log_dir, 'best_model_Q%d_D%d'%(self.n_qubits,self.trainset_depth))
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
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
            avg_rew_train,avg_hilbert_schmidt_dist_train,avg_trace_dist_train=model_evaluation(self.train_circ,self.train_label,self.environment,self.model)
            avg_rew_eval,avg_hilbert_schmidt_dist_eval,avg_trace_dist_eval=model_evaluation(self.val_circ,self.val_label,self.environment,self.model)
            self.eval_results.append([avg_rew_eval,avg_hilbert_schmidt_dist_eval,avg_trace_dist_eval])
            self.train_results.append([avg_rew_train,avg_hilbert_schmidt_dist_train,avg_trace_dist_train])
            self.timestep_list.append(self.num_timesteps)
            # Mean training reward over the last 100 episodes
           
            if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, avg_rew_eval))

                # New best model, you could save the agent here
            if avg_rew_eval > self.best_mean_reward:
                self.best_mean_reward = avg_rew_eval
                # Example for saving best model
                if self.save_best is True:
                    if self.verbose >0:
                        print("Saving new best model at {} timesteps".format(self.num_timesteps))
                        print("Saving new best model to {}.zip".format(self.save_path))
                    self.model.save(self.save_path)
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

        if self.plot is True:
            self.plot_results()

        pass

    def plot_results(self):
        train_results=np.array(self.train_results)
        eval_results=np.array(self.eval_results)
        time_steps=np.array(self.timestep_list)
        
        fig=plt.figure(figsize=(15,5))
        fig.suptitle('Training and evaluation plots', fontsize=15)
        ax=fig.add_subplot(131)
        ax1=fig.add_subplot(132)
        ax2=fig.add_subplot(133)
        plt.subplots_adjust(left=0.065, bottom=None, right=0.971, top=None, wspace=0.27, hspace=None)
        ax.plot(time_steps,eval_results[:,0],label='evaluation_set',marker='x')
        ax.set(xlabel='total_timesteps', ylabel='Reward',title='Average final reward')
        ax1.plot(time_steps,eval_results[:,1],marker='x')
        ax1.set(xlabel='total_timesteps', ylabel='H-S distance',title='Hilbert-Schmidt distance between dm')
        ax2.plot(time_steps,eval_results[:,2],marker='x')
        ax2.set(xlabel='total_timesteps', ylabel='Trace Distance',title='Trace distance between dm')

        ax.plot(time_steps,train_results[:,0],color='orange',label='train_set',marker='x')
        ax1.plot(time_steps,train_results[:,1],color='orange',label='train_set',marker='x')
        ax2.plot(time_steps,train_results[:,2],color='orange',label='train_set',marker='x')
        ax.legend()
        
        fig.savefig(self.plot_dir+'Q%d_D%d_steps%d'%(self.n_qubits,self.trainset_depth,self.timestep_list[-1]))
        plt.show()

    def generalization_test():
        #here will be tested the simple generalization (1 depth for train and different for test)
        # and a more complex one (same depth but test circuit sligthly different noise parameter)
        return 0