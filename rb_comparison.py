import numpy as np
from stable_baselines3 import PPO
from rlnoise.utils import model_evaluation, RB_evaluation
from rlnoise.rewards import DensityMatrixReward
from rlnoise.dataset import CircuitRepresentation

results_list_untrained=[]
results_list_trained = []
result_RB_list = []
model_trained = PPO.load("src/rlnoise/simulation_phase/3Q_random_Clifford/3Q_Rand_clif_logmse340000.zip")

reward = DensityMatrixReward()
rep = CircuitRepresentation()

nqubits=3
n_circuit_in_dataset=50
depth_list=np.arange(3,31,3)
benchmark_circ_path = 'src/rlnoise/simulation_phase/RB/3Q/dataset/'
bench_results_path = 'src/rlnoise/simulation_phase/RB/3Q/results'
result_filename='comparison_results_3Q.npz'
for d in depth_list:
    dataset_name='RB_set'+'_D%d_%dQ_len%d.npz'%(d,nqubits,n_circuit_in_dataset)
    with open(benchmark_circ_path+dataset_name,"rb") as f:
        tmp=np.load(f,allow_pickle=True)
        val_set=tmp['clean_rep']
        val_label=tmp['label']
    results_trained_model = model_evaluation(val_set,val_label,model_trained,reward=reward,representation=rep)
    results_RB = RB_evaluation(lambda_RB=0.08,circ_representation=val_set,target_label=val_label)
    results_list_trained.append(results_trained_model)
    result_RB_list.append(results_RB)
model_results = np.array(results_list_trained)
rb_results = np.array(result_RB_list)

with open(bench_results_path+result_filename,"wb") as f:
    np.savez(f,
             trained=model_results,
             RB=rb_results)


        #    TRAINING ON DIFFERENT DATASET SIZE (Evaluating the best dataset size for overfitting)

# circuits_depth=15                    
# n_circ=[10,50,200,400]
# f = open(benchmark_circ_path+"/depth_%dDep-Term_CZ_3Q_1000.npz"%(circuits_depth),"rb")
# tmp=np.load(f,allow_pickle=True)
# val_set=copy.deepcopy(tmp['val_set'])
# val_label=copy.deepcopy(tmp['val_label'])
# train_set=copy.deepcopy(tmp['train_set'])
# train_label=copy.deepcopy(tmp['train_label'])
# for data_size in n_circ:

#     circuit_env_training = QuantumCircuit(
#     circuits = train_set[:data_size],
#     representation = rep,
#     labels = train_label[:data_size],
#     reward = reward,
#     kernel_size=kernel_size
#     )
#     callback=CustomCallback(check_freq=2000,evaluation_set=tmp,train_environment=circuit_env_training,trainset_depth=circuits_depth,test_on_data_size=data_size)                                          
#     model = PPO(
#     policy,
#     circuit_env_training,
#     policy_kwargs=policy_kwargs, 
#     verbose=0,
#     )
#     model.learn(100000,progress_bar=True, callback=callback)
# f.close()


