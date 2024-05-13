from rlnoise.dataset import Dataset

exp_folder = "simulation/experiments/3q_mixed_dataset/"

config_file = exp_folder + "config.json"
save_path = exp_folder + "dataset"
eval_path = exp_folder + "eval_dataset"

dataset = Dataset(config_file)
dataset.save(save_path)
dataset.generate_eval_dataset(eval_path)
