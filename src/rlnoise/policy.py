from configparser import ConfigParser
import os
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from rlnoise.utils import model_evaluation
import matplotlib.pyplot as plt

params=ConfigParser()
params.read("src/rlnoise/config.ini")
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
        sample = torch.as_tensor(observation_space.sample()[None]).float()
        filter_shape=(1,2)
        conv1 = torch.nn.Conv2d( in_channels=indim,out_channels=64, kernel_size=filter_shape) #adding pooling layer?
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            #shape = torch.nn.functional.avg_pool2d(conv1(sample),kernel_size=filter_shape).shape
            shape = conv1(sample).shape
            
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
    def __init__(self, check_freq,  evaluation_set,train_environment,trainset_depth, verbose=1,test_on_data_size=None):
        super(CustomCallback, self).__init__(verbose)
        self.save_best=params.getboolean('policy','save_best_model')
        self.plot=params.getboolean('policy','plot_results')
        self.best_model_name=params.get('policy','model_name')
        self.plot_name=params.get('policy','plot_name')
        self.log_dir = os.getcwd()+'/src/rlnoise/saved_models/'
        self.plot_dir=os.getcwd()+'/src/rlnoise/data_analysis/plots/'
        self.results_path=os.getcwd()+'/src/rlnoise/bench_results/'
        self.check_freq = check_freq
        self.test_on_data_size=test_on_data_size
        self.environment=train_environment
        self.best_mean_reward = -np.inf
        if self.test_on_data_size is not None:
            self.train_circ=evaluation_set['train_set'][:self.test_on_data_size]
            self.train_label=evaluation_set['train_label'][:self.test_on_data_size]
        else:
            self.train_circ=evaluation_set['train_set']
            self.train_label=evaluation_set['train_label']

        self.val_circ=evaluation_set['val_set']
        self.val_label=evaluation_set['val_label']
        self.dataset_size=len(self.train_circ)
        self.trainset_depth=trainset_depth
        self.n_qubits=self.train_circ[0].shape[1]
        self.eval_results=[]
        self.train_results=[]
        self.timestep_list=[]
        self.save_path = os.path.join(self.log_dir, self.best_model_name)
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
        if self.n_calls==1 or self.n_calls % self.check_freq == 0:

          # Retrieve training reward
            avg_rew_train,avg_hilbert_schmidt_dist_train,avg_trace_dist_train=model_evaluation(self.train_circ,self.train_label,self.environment,self.model)
            avg_rew_eval,avg_hilbert_schmidt_dist_eval,avg_trace_dist_eval=model_evaluation(self.val_circ,self.val_label,self.environment,self.model)
            self.eval_results.append([avg_rew_eval,avg_hilbert_schmidt_dist_eval,avg_trace_dist_eval])
            self.train_results.append([avg_rew_train,avg_hilbert_schmidt_dist_train,avg_trace_dist_train])
            self.timestep_list.append(self.num_timesteps)
           
            if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, avg_rew_eval))

            if avg_rew_eval > self.best_mean_reward:
                self.best_mean_reward = avg_rew_eval
                # Saving best model
                if self.save_best is True:
                    if self.verbose >0:
                        print("Saving new best model at {} timesteps".format(self.num_timesteps))
                        print("Saving new best model to {}.zip".format(self.save_path))
                    self.model.save(self.save_path+str(self.num_timesteps))
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
        self.train_results=np.array(self.train_results)
        self.eval_results=np.array(self.eval_results)
        self.timestep_list=np.array(self.timestep_list)
        if self.plot is True:
            self.plot_results()
        '''
        if self.test_on_data_size is not None:
            f = open(self.results_path+"test_size%d_D_%d_Dep-Term_CZ_3Q.npz"%(self.dataset_size,self.trainset_depth),"wb")
            np.savez(f,train_results=self.train_results, val_results=self.eval_results, timesteps=self.timestep_list)
            f.close()
        '''
        pass 

    def plot_results(self):
        train_results=np.array(self.train_results)
        eval_results=np.array(self.eval_results)
        time_steps=np.array(self.timestep_list)/1000

        if self.test_on_data_size is None:
            fig=plt.figure(figsize=(15,5))
            fig.suptitle('3Q D7 K3 SR-off,Penal=0.001, datasize=100, Coherent(e_z=0.1,e_x=0.2),Std_noise=None', fontsize=15)
            ax=fig.add_subplot(131)
            ax1=fig.add_subplot(132)
            ax2=fig.add_subplot(133)
            plt.subplots_adjust(left=0.065, bottom=None, right=0.971, top=None, wspace=0.27, hspace=None)
            ax.plot(time_steps,eval_results[:,0],label='evaluation_set',marker='x')
            ax.set(xlabel='timesteps/1000', ylabel='Reward',title='Average final reward')
            ax1.plot(time_steps,eval_results[:,1],marker='x')
            ax1.set(xlabel='timesteps/1000', ylabel='H-S distance',title='Hilbert-Schmidt distance between DM')
            ax2.plot(time_steps,eval_results[:,2],marker='x')
            ax2.set(xlabel='timesteps/1000', ylabel='Trace Distance',title='Trace distance between DM')

            ax.plot(time_steps,train_results[:,0],color='orange',label='train_set',marker='x')
            ax1.plot(time_steps,train_results[:,1],color='orange',label='train_set',marker='x')
            ax2.plot(time_steps,train_results[:,2],color='orange',label='train_set',marker='x')
            ax.legend()
            fig.savefig(self.plot_dir+self.plot_name+'_Q%d_D%d_steps%d.png'%(self.n_qubits,self.trainset_depth,self.timestep_list[-1]))
            plt.show()
        else:
            fig=plt.figure(figsize=(5,5))
            fig.suptitle('3Q w CZ D7 K3 SR-off train dataset size=%d, val dataset size=%d'%(self.dataset_size,len(self.val_circ)) , fontsize=15)
            ax=fig.add_subplot(111)
            ax.set(xlabel='timesteps/1000', ylabel='Reward',title='Average final reward')
            ax.plot(time_steps,eval_results[:,0],label='evaluation_set',marker='x')
            ax.plot(time_steps,train_results[:,0],color='orange',label='train_set',marker='x')
            ax.legend()
            fig.savefig(self.plot_dir+'test_size%d_D_%d_Dep-Term_CZ_%dQ'%(self.dataset_size,self.trainset_depth,self.n_qubits))
            #plt.show()


    def generalization_test():
        #here will be tested the simple generalization (1 depth for train and different for test)
        # and a more complex one (same depth but test circuit sligthly different noise parameter)
        return 0