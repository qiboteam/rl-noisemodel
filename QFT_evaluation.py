import numpy as np
# from rlnoise.dataset import CircuitRepresentation
# from rlnoise.rewards.rewards import DensityMatrixReward
# from rlnoise.policy import CNNFeaturesExtractor,CustomCallback
# from rlnoise.gym_env import QuantumCircuit
# from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from qibo.models import QFT
from qibo.transpiler.unroller import Unroller, NativeGates

circuit = QFT(3, with_swaps=False)
natives = NativeGates.U3 | NativeGates.CZ
unroller = Unroller(native_gates = natives)

unrolled_circuit = unroller(circuit)
print(unrolled_circuit.draw())