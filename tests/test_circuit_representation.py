from rlnoise.dataset import load_dataset
import numpy as np
from rlnoise.circuit_representation import CircuitRepresentation

config_file = "tests/config_test.json"

rep = CircuitRepresentation(config_file)
dataset, labels = load_dataset("tests/test_dataset.npz")
circuit = dataset[0]
action_1 = np.asarray([[0, 0, 0, 1],[0, 0, 0, 0],[0, 0, 0, 0]])
action_2 = np.asarray([[0, 0, 0, 0],[1, 1, 1, 1],[0, 0, 0, 0]])

print("Circuit representation:")
print(circuit)

print("Circuit:")
print(rep.rep_to_circuit(circuit).draw())

circuit = circuit.transpose(2,1,0)
print("Action 1:")
print(action_1)
circuit_1 = rep.make_action(action_1, circuit, 0)
# print("Circuit after action 1:")
# print(circuit_1.transpose(2,1,0))
print("Action 2:")
print(action_2)
circuit_2 = rep.make_action(action_2, circuit_1, 1)
# print("Circuit after action 2:")
# print(circuit_2.transpose(2,1,0))

print("Circuit after action 1 and 2:")
circuit = circuit_2.transpose(2,1,0)
print(rep.rep_to_circuit(circuit).draw())

