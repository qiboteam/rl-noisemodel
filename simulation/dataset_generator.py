from rlnoise.dataset import Dataset

config_file = "simulation/experiments/3q_high_noise/config.json"
save_path = "simulation/experiments/3q_high_noise/dataset"

dataset = Dataset(config_file)
dataset.save(save_path)
