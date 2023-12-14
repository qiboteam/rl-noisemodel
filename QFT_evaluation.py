import numpy as np
from rlnoise.dataset import CircuitRepresentation
# from rlnoise.rewards.rewards import DensityMatrixReward
# from rlnoise.policy import CNNFeaturesExtractor,CustomCallback
# from rlnoise.gym_env import QuantumCircuit
# from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from qibo.models import QFT, Circuit
from qibo import gates
from qibo.transpiler.unroller import Unroller, NativeGates

def u3_dec(gate):
    # t, p, l = gate.parameters
    params = gate.parameters
    t = params[0]
    p = params[1]
    l = params[2]
    #print("parameters", params)
    decomposition = []
    if l != 0.0:
        decomposition.append(gates.RZ(gate.qubits[0], l))
    decomposition.append(gates.RX(gate.qubits[0], np.pi/2, 0))
    if t != -np.pi:
        decomposition.append(gates.RZ(gate.qubits[0], t + np.pi))
    decomposition.append(gates.RX(gate.qubits[0], np.pi/2, 0))
    if p != -np.pi:
        decomposition.append(gates.RZ(gate.qubits[0], p + np.pi))
    return decomposition


circuit = QFT(3, with_swaps=False)
natives = NativeGates.U3 | NativeGates.CZ
unroller = Unroller(native_gates = natives)

unrolled_circuit = unroller(circuit)
#print(unrolled_circuit.draw())
queue = unrolled_circuit.queue
final_circuit = Circuit(3)
for gate in queue:
    if isinstance(gate, gates.CZ):
        final_circuit.add(gate)
    elif isinstance(gate, gates.RZ):
        final_circuit.add(gate)
    elif isinstance(gate, gates.U3):
        decomposed = u3_dec(gate)
        for decomposed_gate in decomposed:
            final_circuit.add(decomposed_gate)

#print(final_circuit.draw())



        
