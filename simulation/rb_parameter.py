from rlnoise.randomized_benchmarking import run_rb

config_file = "simulation/experiments/3q_high_noise/config.json"
rb_dataset = "simulation/experiments/3q_high_noise/rb_dataset.npz"

optimal_params = run_rb(rb_dataset, config_file)
print(optimal_params)
