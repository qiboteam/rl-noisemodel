from rlnoise.rl_agent import Agent
from rlnoise.gym_env import QuantumCircuit

dataset_file = "simulation/dataset/3q_high_noise.npz"
config_file = "simulation/config/config_3q_high_noise.json"

env = QuantumCircuit(dataset_file = dataset_file, config_file = config_file)
print("Total circuits:")
print(env.n_circ)
print("Qubits:")
print(env.n_qubits)
print("Training circuits:")
print(env.n_circ_train)

agent = Agent(config_file = config_file, env = env)
agent.train(n_steps = 10000)
                                     




