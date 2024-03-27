from rlnoise.dataset import Dataset

config_file = "simulation/config_3q_high_noise.json"
save_path = "data/simulation/3q/high_noise.npz"

dataset = Dataset(config_file)
dataset.save(save_path, val_split=0.2)
