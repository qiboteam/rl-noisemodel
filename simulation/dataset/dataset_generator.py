from rlnoise.dataset import Dataset

config_file = "simulation/config/config_3q_high_noise.json"
save_path = "simulation/dataset/3q_high_noise.npz"

dataset = Dataset(config_file)
dataset.save(save_path, val_split=0.2)
