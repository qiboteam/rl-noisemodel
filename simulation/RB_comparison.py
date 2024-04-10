import numpy as np
from stable_baselines3 import PPO
from rlnoise.utils import model_evaluation, RB_evaluation
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
