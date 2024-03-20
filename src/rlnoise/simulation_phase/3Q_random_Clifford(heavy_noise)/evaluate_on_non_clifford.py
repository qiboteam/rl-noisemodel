import numpy as np
from rlnoise.dataset import CircuitRepresentation
from rlnoise.rewards import DensityMatrixReward
from stable_baselines3 import PPO
from rlnoise.utils import model_evaluation


config = "src/rlnoise/simulation_phase/1Q_non_clifford/config.json"

agent = PPO.load("src/rlnoise/simulation_phase/1Q/500_circ/1Q_logmse187500.zip")

data = np.load("src/rlnoise/simulation_phase/1Q_non_clifford/Rand_cliff_D7_1Q_len200.npz", allow_pickle=True)

rep = CircuitRepresentation(config)

print(data.files)
non_clifford_circ = data["train_set"]
non_clifford_labels = data["train_label"]

results = model_evaluation(non_clifford_circ, non_clifford_labels, agent
                           , DensityMatrixReward(), rep)

print(f'Fidelity: {results["fidelity"]} std{results["fidelity_std"]}, TD: {results["trace_distance"]} std {results["trace_distance_std"]}, BD: {results["bures_distance"]}, std {results["bures_distance_std"]}')

#3 Qubit models trained on Random Clifford and tested on Non Clifford sets
# Len 500 results : Fidelity: [0.871] std[0.067], TD: [0.304] std [0.083], BD: [0.354], std [0.095]
# Len 400 results: Fidelity: [0.857] std[0.074], TD: [0.32] std [0.089], BD: [0.373], std [0.102]
# Len 300 results: Fidelity: [0.81] std[0.115], TD: [0.375] std [0.122], BD: [0.431], std [0.141]
# Len 200 results: Fidelity: [0.804] std[0.092], TD: [0.384] std [0.098], BD: [0.444], std [0.112]
# Len 100 results: Fidelity: [0.783] std[0.108], TD: [0.393] std [0.107], BD: [0.467], std [0.13]
# Len 50 results: Fidelity: [0.765] std[0.122], TD: [0.414] std [0.117], BD: [0.487], std [0.14]
# Len 20 results: Fidelity: [0.697] std[0.127], TD: [0.462] std [0.112], BD: [0.564], std [0.138]

#1 Qubit model trained on D7 Clifford and tested on Non Clifford sets
# Len 500 results: Fidelity: [0.914] std[0.091], TD: [0.253] std [0.138], BD: [0.264], std [0.143]