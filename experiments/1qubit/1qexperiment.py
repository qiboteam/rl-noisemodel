"""Single-qubit experiment reproducing the original 1q simulation results.

Replicates the dataset generation and RL training from the old scripts
(old/scripts/dataset_generator.py and old/scripts/model_train.py) using
the new API, with the exact same parameters from experiments/simulation/1q/config.json:

  dataset  : 100 Clifford circuits, 10 moments, 1 qubit
  noise    : depolarizing (lam=0.02) on rz,
             amplitude damping (p0=0.03) on rx,
             coherent-x (eps=0.04) on rx,
             coherent-z (eps=0.02) on rz
  agent    : MlpPolicy, features_dim=32, filter_size=1, n_filters=16,
             n_steps=1000, batch_size=200,  net_arch=[32,32]
  training : 200 000 total timesteps, eval every 2 000 steps
"""

from pathlib import Path

import qibo

qibo.set_backend("numpy")

from rlnoise.circuit_encoder import CircuitEncoder
from rlnoise.config import (
    AgentConfig,
    DatasetConfig,
    GateSpecificNoise,
    GymEnvConfig,
    NoiseConfig,
    RewardConfig,
)
from rlnoise.dataset import DatasetGenerator
from rlnoise.gym_env import QuantumCircuitEnv
from rlnoise.rl_agent import RLAgent

# ---------------------------------------------------------------------------
# Paths  (all relative to the project root, regardless of where the script
# is invoked from)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
EXP_FOLDER = PROJECT_ROOT / "experiments" / "1qubit"
EXP_FOLDER.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Dataset configuration  (config.json → "dataset" section)
# ---------------------------------------------------------------------------
dataset_config = DatasetConfig(
    n_circuits=100,
    moments=10,
    qubits=1,
    primitive_gates=["rz", "rx"],
    clifford=True,
    distributed_clifford=False,
    mixed=False,
)

# ---------------------------------------------------------------------------
# 2. Noise configuration  (config.json → "noise" section)
#
#   dep_lambda  = 0.02  on rz  (depol_on_gate)
#   p0          = 0.03  on rx  (damping_on_gate)
#   epsilon_x   = 0.04  on rx  (x_coherent_on_gate)
#   epsilon_z   = 0.02  on rz  (z_coherent_on_gate)
# ---------------------------------------------------------------------------
noise_config = NoiseConfig(
    noise_list=[
        GateSpecificNoise(gate="rz", noise_channel="depolarizing",  noise_parameter=0.02),
        GateSpecificNoise(gate="rx", noise_channel="damping",        noise_parameter=0.03),
        GateSpecificNoise(gate="rx", noise_channel="coherent_x",     noise_parameter=0.04),
        GateSpecificNoise(gate="rz", noise_channel="coherent_z",     noise_parameter=0.02),
    ]
)

print(noise_config)

# ---------------------------------------------------------------------------
# 3. Generate training dataset  (n_circuits=100, moments=10)
# ---------------------------------------------------------------------------
print("\n--- Generating training dataset ---")
generator = DatasetGenerator(dataset_config, noise_config)
dataset = generator.generate(verbose=True)
dataset.save(str(EXP_FOLDER / "dataset"))
print(dataset)

# ---------------------------------------------------------------------------
# 4. Generate evaluation dataset  (eval_size=100, eval_depth=15)
# ---------------------------------------------------------------------------
print("\n--- Generating evaluation dataset ---")
eval_config = DatasetConfig(
    n_circuits=100,
    moments=15,           # eval_depth = 15
    qubits=1,
    primitive_gates=["rz", "rx"],
    clifford=True,
    distributed_clifford=False,
    mixed=False,
)
eval_generator = DatasetGenerator(eval_config, noise_config)
eval_dataset = eval_generator.generate(verbose=True)
eval_dataset.save(str(EXP_FOLDER / "eval_dataset"))
print(eval_dataset)

# ---------------------------------------------------------------------------
# 5. Gym environment  (config.json → "gym_env" and "reward" sections)
# ---------------------------------------------------------------------------
env_config = GymEnvConfig(
    kernel_size=3,
    action_penalty=0.0,
    action_space_max_value=0.06,
    enable_only_depolarizing=False,
    val_split=0.2,
)

reward_config = RewardConfig(
    metric="trace",
    function="inverted_squared",
    alpha=20,
)

encoder = CircuitEncoder(primitive_gates=["rz", "rx"])

env = QuantumCircuitEnv(
    dataset=dataset,
    encoder=encoder,
    env_config=env_config,
    reward_config=reward_config,
)

print("\n", env)

# ---------------------------------------------------------------------------
# 6. RL Agent  (config.json → "agent" section)
#
#   policy          = MlpPolicy
#   features_dim    = 32
#   filter_size     = 1   (conv_layers=1 → single Conv2D)
#   n_filters       = 16
#   net_arch        = pi=[32,32], vf=[32,32]   (same defaults as old code)
#   nn_update_steps = 1000  → n_steps
#   batch_size      = 200
# ---------------------------------------------------------------------------
agent_config = AgentConfig(
    policy="MlpPolicy",
    features_dim=32,
    filter_size=1,
    n_filters=16,
    pi_net_arch=[32, 32],
    vf_net_arch=[32, 32],
    n_steps=1000,
    batch_size=200,
)

agent = RLAgent(env=env, agent_config=agent_config)
print(agent)

# ---------------------------------------------------------------------------
# 7. Training  (total_timesteps=200000, check_freq=2000)
# ---------------------------------------------------------------------------
save_path = str(EXP_FOLDER / "model_1")

print("\n--- Training ---")
results = agent.train(
    total_timesteps=200000,
    check_freq=2000,
    save_path=save_path,
    save_best=True,
    progress_bar=True,
    verbose=True,
)

print(f"\nTraining complete. Best model saved to: {save_path}")
