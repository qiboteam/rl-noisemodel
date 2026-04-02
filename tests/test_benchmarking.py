"""Tests for benchmarking utilities."""

import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing

from rlnoise.benchmarking import (
    maximally_mixed_state,
    generate_rb_circuits,
    fit_rb_decay,
    _build_composite_circuit,
    _apply_rb_noise_model,
)
from rlnoise.config import DatasetConfig, NoiseConfig, GateSpecificNoise
from rlnoise.circuit_generator import CircuitGenerator
from rlnoise.circuit_encoder import CircuitEncoder
from rlnoise.noise_model import QuantumNoiseModel


class TestMaximallyMixedState:
    """Tests for the maximally_mixed_state helper."""

    def test_qubit_1(self):
        mms = maximally_mixed_state(2)
        assert mms.shape == (2, 2)
        assert np.allclose(mms, np.eye(2) / 2)

    def test_qubit_2(self):
        mms = maximally_mixed_state(4)
        assert mms.shape == (4, 4)
        assert np.allclose(np.trace(mms), 1.0)

    def test_complex_dtype(self):
        mms = maximally_mixed_state(2)
        assert np.iscomplexobj(mms)


class TestBuildCompositeCircuit:
    """Tests for _build_composite_circuit."""

    def test_doubles_circuit_depth(self):
        from qibo.models import Circuit
        from qibo import gates
        circuit = Circuit(1, density_matrix=True)
        circuit.add(gates.RX(0, theta=np.pi / 4))
        composite = _build_composite_circuit(circuit)
        # Composite should have at least twice the gates
        assert len(composite.queue) >= 2

    def test_returns_same_nqubits(self):
        from qibo.models import Circuit
        from qibo import gates
        circuit = Circuit(2, density_matrix=True)
        circuit.add(gates.RX(0, theta=0.1))
        circuit.add(gates.RZ(1, theta=0.2))
        composite = _build_composite_circuit(circuit)
        assert composite.nqubits == 2


class TestApplyRbNoiseModel:
    """Tests for _apply_rb_noise_model."""

    def test_returns_circuit(self):
        from qibo.models import Circuit
        from qibo import gates
        circuit = Circuit(1, density_matrix=True)
        circuit.add(gates.RX(0, theta=0.1))
        noisy = _apply_rb_noise_model(circuit, lambda_rb=0.9)
        assert noisy is not None

    def test_perfect_lambda_no_noise(self):
        from qibo.models import Circuit
        from qibo import gates
        circuit = Circuit(1, density_matrix=True)
        circuit.add(gates.RX(0, theta=0.1))
        # lambda_rb=1 → p=0, no depolarizing noise
        noisy = _apply_rb_noise_model(circuit, lambda_rb=1.0)
        assert noisy is not None


@pytest.fixture
def rb_setup():
    """Create components needed for RB tests."""
    config = DatasetConfig(
        n_circuits=3,
        qubits=1,
        moments=4,
        clifford=True,
        primitive_gates=["rx", "rz"],
    )
    noise_config = NoiseConfig(
        noise_list=[
            GateSpecificNoise(gate="rx", noise_channel="depolarizing", noise_parameter=0.01),
            GateSpecificNoise(gate="rz", noise_channel="depolarizing", noise_parameter=0.01),
        ]
    )
    circuit_gen = CircuitGenerator(config)
    encoder = CircuitEncoder(primitive_gates=["rx", "rz"])
    noise_model = QuantumNoiseModel(noise_config, qubits=1)
    return circuit_gen, encoder, noise_model


class TestGenerateRbCircuits:
    """Tests for generate_rb_circuits."""

    def test_returns_list_per_depth(self, rb_setup):
        circuit_gen, encoder, noise_model = rb_setup
        depths = [3, 5]
        rb_data = generate_rb_circuits(
            circuit_gen, encoder, noise_model,
            depths=depths, n_circuits_per_depth=2
        )
        assert len(rb_data) == len(depths)

    def test_entry_structure(self, rb_setup):
        circuit_gen, encoder, noise_model = rb_setup
        rb_data = generate_rb_circuits(
            circuit_gen, encoder, noise_model,
            depths=[4], n_circuits_per_depth=2
        )
        entry = rb_data[0]
        assert "depth" in entry
        assert "circuit_arrays" in entry
        assert "qibo_circuits" in entry
        assert "labels" in entry
        assert entry["depth"] == 4
        assert len(entry["qibo_circuits"]) == 2

    def test_labels_shape(self, rb_setup):
        circuit_gen, encoder, noise_model = rb_setup
        rb_data = generate_rb_circuits(
            circuit_gen, encoder, noise_model,
            depths=[3], n_circuits_per_depth=2
        )
        labels = rb_data[0]["labels"]
        # 1-qubit → 2x2 density matrices
        assert labels.shape[1:] == (2, 2)

    def test_restores_original_moments(self, rb_setup):
        circuit_gen, encoder, noise_model = rb_setup
        original = circuit_gen.n_moments
        generate_rb_circuits(
            circuit_gen, encoder, noise_model,
            depths=[6], n_circuits_per_depth=1
        )
        assert circuit_gen.n_moments == original


class TestFitRbDecay:
    """Tests for fit_rb_decay."""

    def test_returns_tuple(self, rb_setup):
        circuit_gen, encoder, noise_model = rb_setup
        rb_data = generate_rb_circuits(
            circuit_gen, encoder, noise_model,
            depths=[3, 5, 7], n_circuits_per_depth=2
        )
        a, lam = fit_rb_decay(rb_data, noise_model)
        assert isinstance(a, float)
        assert isinstance(lam, float)

    def test_lambda_in_valid_range(self, rb_setup):
        circuit_gen, encoder, noise_model = rb_setup
        rb_data = generate_rb_circuits(
            circuit_gen, encoder, noise_model,
            depths=[3, 5, 7], n_circuits_per_depth=2
        )
        _, lam = fit_rb_decay(rb_data, noise_model)
        assert 0.0 <= lam <= 1.0
