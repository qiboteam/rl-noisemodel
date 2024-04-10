from tqdm import tqdm
from scipy.optimize import curve_fit
from qibo import gates, Circuit
import numpy as np


def RB_evaluation(lambda_RB,circ_representation,target_label):
    dataset_size = len(target_label)
    trace_distance_rb_list = []
    bures_distance_rb_list = []
    fidelity_rb_list = []
    trace_distance_no_noise_list = []
    bures_distance_no_noise_list = []
    fidelity_no_noise_list = []
    rb_noise_model=CustomNoiseModel("src/rlnoise/config.json")
    RB_label = np.array([rb_noise_model.apply(CircuitRepresentation().rep_to_circuit(circ_representation[i]))().state() 
                         for i in range(dataset_size)])
    label_no_noise_added = np.array([CircuitRepresentation().rep_to_circuit(circ_representation[i])().state() 
                         for i in range(dataset_size)])
    for idx,label in enumerate(RB_label):
        fidelity_rb_list.append(compute_fidelity(label,target_label[idx]))
        trace_distance_rb_list.append(trace_distance(label,target_label[idx]))
        bures_distance_rb_list.append(bures_distance(label,target_label[idx]))
        fidelity_no_noise_list.append(compute_fidelity(label_no_noise_added[idx],target_label[idx]))
        trace_distance_no_noise_list.append(trace_distance(label_no_noise_added[idx],target_label[idx]))
        bures_distance_no_noise_list.append(bures_distance(label_no_noise_added[idx],target_label[idx]))
    fidelity = np.array(fidelity_rb_list)
    trace_dist = np.array(trace_distance_rb_list)
    bures_dist = np.array(bures_distance_rb_list)
    no_noise_fidelity = np.array(fidelity_no_noise_list)
    no_noise_trace_dist = np.array(trace_distance_no_noise_list)
    no_noise_bures_dist = np.array(bures_distance_no_noise_list)
    results = np.array([(
                       fidelity.mean(),fidelity.std(),
                       trace_dist.mean(),trace_dist.std(),
                       bures_dist.mean(),bures_dist.std(),
                       no_noise_fidelity.mean(),no_noise_fidelity.std(),
                       no_noise_trace_dist.mean(),no_noise_trace_dist.std(),
                       no_noise_bures_dist.mean(),no_noise_bures_dist.std()  )],
                       dtype=[
                              ('fidelity','<f4'),('fidelity_std','<f4'),
                              ('trace_distance','<f4'),('trace_distance_std','<f4'),
                              ('bures_distance','<f4'),('bures_distance_std','<f4'),
                              ('fidelity_no_noise','<f4'),('fidelity_no_noise_std','<f4'),
                              ('trace_distance_no_noise','<f4'),('trace_distance_no_noise_std','<f4'),
                              ('bures_distance_no_noise','<f4'),('bures_distance_no_noise_std','<f4')  ])
    
    return results
