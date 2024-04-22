from rlnoise.randomized_benchmarking import rb_dataset_generator

exp_folder = "simulation/experiments/3q_low_noise/"

config_file = exp_folder + "config.json"

rb_dataset_generator(config_file)