from rlnoise.gym_env import QuantumCircuit
from rlnoise.circuit_representation import CircuitRepresentation
import numpy as np

config_file = "tests/config_test.json"
dataset_file = "tests/test_dataset.npz"

env = QuantumCircuit(config_file, dataset_file)
print("Total circuits:")
print(env.n_circ)
print("Qubits:")
print(env.n_qubits)
print("Training circuits:")
print(env.n_circ_train)
print("Observation space:")
print(env.observation_space)
print("Action space:")
print(env.action_space)

rep = CircuitRepresentation(config_file)
action = np.asarray([[0, 0, 0, 1],[0, 0, 0, 0],[0, 0, 0, 0]])

print("First circuit:")
print(rep.rep_to_circuit(env.circuits[0]).draw())
print("Reset:")
reset_result, reset_info = env.reset(i=0)
print("Reset result:")  
print(reset_result.transpose(2,1,0))
print("Action:")
print(action)
obs, reward, term, _, __ = env.step(action)
print("Observation after action 1:")
print(obs.transpose(2,1,0))
assert reward == 0., "Error in reward"
assert not term, "Error in termination"

print("Termination")
for i in range(env.circuit_lenght - 2):
    obs, reward, term, _, __ = env.step(action)
    assert reward == 0., "Error in reward"
    assert not term, "Error in termination"

obs, reward, term, _, __ = env.step(action)
print("Reward: ", reward)
assert term, "Error in termination"

print("Final circuit: ")
print(env.get_qibo_circuit().draw())