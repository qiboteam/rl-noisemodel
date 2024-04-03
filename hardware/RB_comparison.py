import numpy as np
from stable_baselines3 import PPO
from rlnoise.utils import model_evaluation, RB_evaluation
from rlnoise.dataset import CircuitRepresentation

results_list_untrained=[]
results_list_trained = []
result_RB_list = []
model_trained = PPO.load("src/rlnoise/hardware_test/trained_models/1Q_d11_best/1Q_D11_hardware_clip0.12_MIT_177500.zip")

reward = DensityMatrixReward()
rep = CircuitRepresentation()

bench_results_path = 'src/rlnoise'
result_filename='rb_hardware_q2.npz'

n_circuit_in_dataset=20
benchmark_circ_path = "src/rlnoise/hardware_test/RB/dataset/"

depth_list=np.arange(1,142,20)
for d in depth_list:
    print(d)
    dataset_name='RB_set_D%d_1Q_len%d_circs_result.npy'%(d,n_circuit_in_dataset)
    with open(benchmark_circ_path+dataset_name,"rb") as f:
        tmp=np.load(f,allow_pickle=True)
        circ=tmp[:,0]
        val_label=tmp[:,2]
    val_set = np.array([CircuitRepresentation().circuit_to_array(circuit) for circuit in circ])

    results_trained_model = model_evaluation(val_set,val_label,model_trained,reward=reward,representation=rep)
    results_RB = RB_evaluation(lambda_RB=0.045,circ_representation=val_set,target_label=val_label)
    results_list_trained.append(results_trained_model)
    result_RB_list.append(results_RB)
model_results = np.array(results_list_trained)
rb_results = np.array(result_RB_list)

with open(bench_results_path+result_filename,"wb") as f:
    np.savez(f,
             trained=model_results,
             RB=rb_results)