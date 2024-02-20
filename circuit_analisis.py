from qibo import gates, Circuit

def load_dataset():
    return None

dataset = load_dataset()

damping = 0
dep = 0

for circuit in dataset:
    for gate in circuit.queue:
        if isinstance(gate, gates.AmplitudeDampingChannel):
            damping += 1
        elif isinstance(gate, gates.DepolarizingChannel):
            dep += 1

tot_channels = damping + dep
print(f"Total number of channels: {tot_channels}")
print(f"Number of amplitude damping channels: {damping}")
print(f"Number of depolarizing channels: {dep}")
print(f"Fraction of amplitude damping channels: {damping/tot_channels}")
print(f"Fraction of depolarizing channels: {dep/tot_channels}")

