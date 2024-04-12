from rlnoise.randomized_benchmarking import run_rb, rb_evaluation
import numpy as np

config_file = "simulation/experiments/3q_high_noise/config.json"
rb_dataset = "simulation/experiments/3q_high_noise/rb_dataset.npz"
result_file = "simulation/experiments/3q_high_noise/rb_result.npz"

optimal_params = run_rb(rb_dataset, config_file)
print("RB Model:")
print(optimal_params)
decay_constant = 1 - optimal_params["l"]    
print("Decay constant: ", decay_constant)

result = rb_evaluation(decay_constant, rb_dataset, config_file)
print("Result of RB evaluation, the columns are: depth, fidelity, fidelity_std, trace_distance, trace_distance_std, fidelity_no_noise, fidelity_no_noise_std, trace_distance_no_noise, trace_distance_no_noise_std")
print(result)

np.savez(result_file, result=result)

