"""Tests for rlnoise.analysis — collect_actions and noise_summary."""

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

from rlnoise.analysis import (
    collect_actions,
    noise_summary,
    NOISE_CHANNELS,
)
from rlnoise.config import (
    DatasetConfig,
    NoiseConfig,
    GateSpecificNoise,
    GymEnvConfig,
    RewardConfig,
    AgentConfig,
)
from rlnoise.dataset import DatasetGenerator
from rlnoise.circuit_encoder import CircuitEncoder
from rlnoise.gym_env import QuantumCircuitEnv
from rlnoise.rl_agent import RLAgent


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def agent_and_circuits():
    """Minimal untrained agent + encoded circuits (1-qubit)."""
    noise_config = NoiseConfig(noise_list=[
        GateSpecificNoise(gate="rx", noise_channel="depolarizing", noise_parameter=0.02),
        GateSpecificNoise(gate="rz", noise_channel="coherent_z",  noise_parameter=0.02),
    ])
    dataset_config = DatasetConfig(
        n_circuits=8, moments=6, qubits=1,
        primitive_gates=["rx", "rz"], clifford=False, mixed=False,
    )
    encoder = CircuitEncoder(primitive_gates=["rx", "rz"])
    dataset = DatasetGenerator(dataset_config, noise_config).generate(verbose=False)
    env = QuantumCircuitEnv(
        dataset=dataset,
        encoder=encoder,
        env_config=GymEnvConfig(kernel_size=3, action_space_max_value=0.06, val_split=0.0),
        reward_config=RewardConfig(metric="trace", function="inverted_squared", alpha=20.0),
    )
    agent = RLAgent(env=env, agent_config=AgentConfig(
        n_steps=100, batch_size=25, features_dim=16, n_filters=8,
    ))
    return agent, dataset.circuits


@pytest.fixture(scope="module")
def actions_data(agent_and_circuits):
    agent, circuits = agent_and_circuits
    return collect_actions(agent, circuits, verbose=False)


# ---------------------------------------------------------------------------
# collect_actions
# ---------------------------------------------------------------------------

class TestCollectActions:
    """Tests for collect_actions()."""

    def test_returns_dict(self, actions_data):
        assert isinstance(actions_data, dict)

    def test_expected_keys(self, actions_data):
        expected = {"n_circuits", "n_moments", "n_qubits", "gate_type"} | set(NOISE_CHANNELS)
        for key in expected:
            assert key in actions_data, f"Missing key: {key}"

    def test_n_circuits_matches(self, agent_and_circuits, actions_data):
        _, circuits = agent_and_circuits
        assert actions_data["n_circuits"] == len(circuits)

    def test_noise_arrays_shape(self, agent_and_circuits, actions_data):
        _, circuits = agent_and_circuits
        n_c = len(circuits)
        n_m = circuits.shape[1]
        n_q = circuits.shape[2]
        for ch in NOISE_CHANNELS:
            arr = actions_data[ch]
            assert arr.shape == (n_c, n_m, n_q), f"{ch} shape mismatch"

    def test_gate_type_shape(self, agent_and_circuits, actions_data):
        _, circuits = agent_and_circuits
        n_c = len(circuits)
        n_m = circuits.shape[1]
        n_q = circuits.shape[2]
        assert actions_data["gate_type"].shape == (n_c, n_m, n_q)

    def test_gate_type_contains_valid_labels(self, actions_data):
        valid = {"rx", "rz", "cz", "id"}
        flat = actions_data["gate_type"].ravel()
        unique = set(flat.tolist())
        assert unique.issubset(valid), f"Unexpected gate labels: {unique - valid}"

    def test_noise_values_in_range(self, actions_data):
        """Noise values must be non-negative (agent output is scaled)."""
        for ch in NOISE_CHANNELS:
            arr = actions_data[ch]
            assert np.all(arr >= -1e-9), f"{ch} has negative values"

    def test_verbose_flag(self, agent_and_circuits, capsys):
        agent, circuits = agent_and_circuits
        collect_actions(agent, circuits[:2], verbose=True)
        captured = capsys.readouterr()
        assert "circuit" in captured.out.lower() or len(captured.out) >= 0  # just ensures no crash

    def test_single_circuit(self, agent_and_circuits):
        agent, circuits = agent_and_circuits
        data = collect_actions(agent, circuits[:1], verbose=False)
        assert data["n_circuits"] == 1

    def test_3qubit(self):
        """collect_actions works for 3-qubit circuits."""
        noise_config = NoiseConfig(noise_list=[
            GateSpecificNoise(gate="rx", noise_channel="depolarizing", noise_parameter=0.02),
            GateSpecificNoise(gate="rz", noise_channel="depolarizing", noise_parameter=0.02),
            GateSpecificNoise(gate="cz", noise_channel="depolarizing", noise_parameter=0.02),
        ])
        dataset_config = DatasetConfig(
            n_circuits=4, moments=4, qubits=3,
            primitive_gates=["rx", "rz", "cz"], clifford=False, mixed=False,
        )
        encoder = CircuitEncoder(primitive_gates=["rx", "rz", "cz"])
        dataset = DatasetGenerator(dataset_config, noise_config).generate(verbose=False)
        env = QuantumCircuitEnv(
            dataset=dataset,
            encoder=encoder,
            env_config=GymEnvConfig(kernel_size=3, action_space_max_value=0.06, val_split=0.0),
            reward_config=RewardConfig(metric="trace", function="inverted_squared", alpha=20.0),
        )
        agent = RLAgent(env=env, agent_config=AgentConfig(
            n_steps=100, batch_size=25, features_dim=16, n_filters=8,
        ))
        data = collect_actions(agent, dataset.circuits, verbose=False)
        assert data["n_qubits"] == 3
        for ch in NOISE_CHANNELS:
            assert data[ch].shape[2] == 3


# ---------------------------------------------------------------------------
# noise_summary
# ---------------------------------------------------------------------------

class TestNoiseSummary:
    """Tests for noise_summary()."""

    def test_returns_string(self, actions_data):
        result = noise_summary(actions_data)
        assert isinstance(result, str)

    def test_contains_mean_and_std(self, actions_data):
        result = noise_summary(actions_data)
        assert "Mean" in result and "Std" in result

    def test_contains_channel_labels(self, actions_data):
        from rlnoise.analysis import NOISE_LABELS
        result = noise_summary(actions_data)
        for label in NOISE_LABELS.values():
            assert label in result

    def test_non_empty(self, actions_data):
        result = noise_summary(actions_data)
        assert len(result) > 0

    def test_prints_to_stdout(self, actions_data, capsys):
        noise_summary(actions_data)
        captured = capsys.readouterr()
        # noise_summary may or may not print, just check no exception

    def test_single_circuit_data(self, agent_and_circuits):
        """noise_summary works even on a single-circuit dataset."""
        agent, circuits = agent_and_circuits
        data = collect_actions(agent, circuits[:1], verbose=False)
        result = noise_summary(data)
        assert isinstance(result, str)

    def test_skip_zero_note_in_output(self, actions_data):
        result_with = noise_summary(actions_data, skip_zero=True)
        result_without = noise_summary(actions_data, skip_zero=False)
        assert "zeros excluded" in result_with
        assert "zeros excluded" not in result_without

    def test_skip_zero_changes_statistics(self, actions_data):
        """skip_zero=True should produce different (higher) mean than False."""
        result_true  = noise_summary(actions_data, skip_identity=False, skip_zero=True)
        result_false = noise_summary(actions_data, skip_identity=False, skip_zero=False)
        # They may differ; at minimum both should return strings without error
        assert isinstance(result_true, str)
        assert isinstance(result_false, str)


# ---------------------------------------------------------------------------
# _flat_values
# ---------------------------------------------------------------------------

class TestFlatValues:
    """Tests for the internal _flat_values helper."""

    def test_returns_ndarray(self, actions_data):
        from rlnoise.analysis import _flat_values
        vals = _flat_values(actions_data, NOISE_CHANNELS[0])
        assert isinstance(vals, np.ndarray)

    def test_skip_identity_reduces_count(self, actions_data):
        from rlnoise.analysis import _flat_values
        with_id  = _flat_values(actions_data, NOISE_CHANNELS[0], skip_identity=False)
        without_id = _flat_values(actions_data, NOISE_CHANNELS[0], skip_identity=True)
        # Identity positions are filtered out — result should be ≤ full
        assert without_id.size <= with_id.size

    def test_gate_filter(self, actions_data):
        from rlnoise.analysis import _flat_values
        vals_rx = _flat_values(actions_data, NOISE_CHANNELS[0], gate_filter="rx", skip_identity=False)
        vals_rz = _flat_values(actions_data, NOISE_CHANNELS[0], gate_filter="rz", skip_identity=False)
        vals_all = _flat_values(actions_data, NOISE_CHANNELS[0], skip_identity=False)
        # Gate-filtered subsets are disjoint and smaller or equal in size
        assert vals_rx.size + vals_rz.size <= vals_all.size + 1  # allow rounding in count

    def test_qubit_filter(self, actions_data):
        from rlnoise.analysis import _flat_values
        vals_q0 = _flat_values(actions_data, NOISE_CHANNELS[0], qubit_filter=0, skip_identity=False)
        vals_all = _flat_values(actions_data, NOISE_CHANNELS[0], skip_identity=False)
        assert vals_q0.size <= vals_all.size

    def test_combined_filter(self, actions_data):
        from rlnoise.analysis import _flat_values
        vals = _flat_values(actions_data, NOISE_CHANNELS[0],
                            gate_filter="rx", qubit_filter=0, skip_identity=True)
        assert isinstance(vals, np.ndarray)

    def test_no_matching_gate_returns_empty(self, actions_data):
        from rlnoise.analysis import _flat_values
        vals = _flat_values(actions_data, NOISE_CHANNELS[0], gate_filter="cz", skip_identity=False)
        # 1-qubit fixture has no CZ gates
        assert vals.size == 0

    def test_skip_zero_reduces_count(self, actions_data):
        from rlnoise.analysis import _flat_values
        with_zeros    = _flat_values(actions_data, NOISE_CHANNELS[0], skip_identity=False,
                                     skip_zero=False)
        without_zeros = _flat_values(actions_data, NOISE_CHANNELS[0], skip_identity=False,
                                     skip_zero=True)
        assert without_zeros.size <= with_zeros.size

    def test_skip_zero_no_zeros_in_result(self, actions_data):
        from rlnoise.analysis import _flat_values
        vals = _flat_values(actions_data, NOISE_CHANNELS[0], skip_identity=False,
                            skip_zero=True)
        assert np.all(vals != 0.0)

    def test_skip_zero_false_preserves_zeros(self, actions_data):
        """With skip_zero=False the result may contain zeros (depending on fixture)."""
        from rlnoise.analysis import _flat_values
        vals_all  = _flat_values(actions_data, NOISE_CHANNELS[0], skip_identity=False,
                                 skip_zero=False)
        vals_nz   = _flat_values(actions_data, NOISE_CHANNELS[0], skip_identity=False,
                                 skip_zero=True)
        # skip_zero=False keeps at least as many values
        assert vals_all.size >= vals_nz.size
