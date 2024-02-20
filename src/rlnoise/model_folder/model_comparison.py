import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from rlnoise.utils import model_evaluation, RL_NoiseModel
from rlnoise.custom_noise import CustomNoiseModel
from rlnoise.metrics import compute_fidelity
from rlnoise.dataset import CircuitRepresentation
from qibo import gates, Circuit
from qibo.noise import NoiseModel
# non_cliff_set = np.load(f"{Path(__file__).parents[1]}/simulation_phase/3Q_non_clifford/non_clifford_set.npz", allow_pickle=True)
# test_circ = non_cliff_set["train_circ"]
# test_labels = non_cliff_set["train_label"]


model_path = "src/rlnoise/model_folder/Rans_cliff_mse_tanh/3Q_Rand_clif_mse_tanh12000.zip"
agent = PPO.load(model_path)



# evaluation_results = model_evaluation(test_circ, test_labels, agent)

# print(f"Fid:{evaluation_results['fidelity']}, Fid_std:{evaluation_results['fidelity_std']}, TD:{evaluation_results['trace_distance']}, TD_std:{evaluation_results['trace_distance_std']}")

# Old_random_generator: Fid:[0.969], Fid_std:[0.014], TD:[0.148], TD_std:[0.032]
# Rand_cliff_generator: Fid:[0.973], Fid_std:[0.017], TD:[0.136], TD_std:[0.04]

# Old_random_generator + Tanh: Fid:[0.972], Fid_std:[0.012], TD:[0.14], TD_std:[0.032]
# Rand_cliff_generator + Tanh: Fid:[0.973], Fid_std:[0.012], TD:[0.137], TD_std:[0.031]

