"""
Golden Physics Tests — Day 9

These tests validate the physics invariants of the noise model layer.
Run with:  python -m pytest tests/test_physics_engine.py -v
"""

from __future__ import annotations

import math
import pytest
from dataclasses import FrozenInstanceError

from qec_noise_factory.physics.pauli_noise import PauliChannel, PauliNoiseModel
from qec_noise_factory.physics.noise_compiler import (
    compile_to_phenomenological,
    compile_to_circuit_level,
    compile_to_pauli_channel_1,
    model_from_preset,
)


# =========================================================================
# PauliChannel tests
# =========================================================================

class TestPauliChannel:

    def test_channel_probabilities_sum(self):
        """px + py + pz must be <= 1.0."""
        ch = PauliChannel(px=0.1, py=0.2, pz=0.3)
        assert ch.total_error() == pytest.approx(0.6)
        assert ch.pi == pytest.approx(0.4)

    def test_channel_rejects_negative(self):
        """Negative probabilities must raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            PauliChannel(px=-0.1, py=0.0, pz=0.0)

    def test_channel_rejects_sum_above_one(self):
        """Total probability > 1.0 must raise ValueError."""
        with pytest.raises(ValueError, match="exceeds 1.0"):
            PauliChannel(px=0.4, py=0.4, pz=0.4)

    def test_channel_frozen(self):
        """PauliChannel must be immutable (frozen)."""
        ch = PauliChannel(px=0.1, py=0.1, pz=0.1)
        with pytest.raises(FrozenInstanceError):
            ch.px = 0.5  # type: ignore

    def test_clean_channel(self):
        """Default channel should have zero error."""
        ch = PauliChannel()
        assert ch.total_error() == 0.0
        assert ch.pi == 1.0

    def test_canonical_tuple_stability(self):
        """canonical_tuple must produce stable, rounded values."""
        ch = PauliChannel(px=1/3, py=1/3, pz=1/3)
        t = ch.canonical_tuple()
        assert len(t) == 3
        # Should be rounded to 12 digits
        assert t == (round(1/3, 12), round(1/3, 12), round(1/3, 12))


# =========================================================================
# PauliChannel utility tests
# =========================================================================

class TestChannelUtilities:

    def test_scale(self):
        """scale(2) should double all probabilities."""
        ch = PauliChannel(px=0.1, py=0.1, pz=0.1)
        scaled = ch.scale(2.0)
        assert scaled.px == pytest.approx(0.2)
        assert scaled.py == pytest.approx(0.2)
        assert scaled.pz == pytest.approx(0.2)

    def test_scale_clamps_at_one(self):
        """Scaling beyond sum=1 should proportionally clamp."""
        ch = PauliChannel(px=0.3, py=0.3, pz=0.3)  # total = 0.9
        scaled = ch.scale(2.0)  # would be 1.8 → clamped
        assert scaled.total_error() == pytest.approx(1.0, abs=1e-9)
        # Proportions should be preserved
        assert scaled.px == pytest.approx(scaled.py)
        assert scaled.py == pytest.approx(scaled.pz)

    def test_scale_zero(self):
        """scale(0) should produce a clean channel."""
        ch = PauliChannel(px=0.1, py=0.2, pz=0.3)
        scaled = ch.scale(0.0)
        assert scaled.total_error() == 0.0

    def test_clamp(self):
        """clamp() should produce a valid channel."""
        # Constructing invalid channel directly isn't possible (validation),
        # so we test that a valid channel clamped is unchanged
        ch = PauliChannel(px=0.1, py=0.2, pz=0.3)
        clamped = ch.clamp()
        assert clamped.px == pytest.approx(0.1)
        assert clamped.py == pytest.approx(0.2)
        assert clamped.pz == pytest.approx(0.3)


# =========================================================================
# PauliNoiseModel preset tests
# =========================================================================

class TestModelPresets:

    def test_symmetric_depolarizing(self):
        """Symmetric model: px = py = pz = p/3."""
        model = PauliNoiseModel.from_symmetric_depolarizing(p=0.03, meas_flip=0.01)
        ch = model.data_noise
        assert ch.px == pytest.approx(0.01)
        assert ch.py == pytest.approx(0.01)
        assert ch.pz == pytest.approx(0.01)
        assert ch.total_error() == pytest.approx(0.03)
        assert model.meas_flip == pytest.approx(0.01)

    def test_biased_z_ratios(self):
        """Biased-Z: Pz/Px = bias, total = p."""
        bias = 100.0
        model = PauliNoiseModel.from_biased_z(p=0.1, bias=bias)
        ch = model.data_noise
        assert ch.total_error() == pytest.approx(0.1)
        assert ch.pz / ch.px == pytest.approx(bias)
        assert ch.px == pytest.approx(ch.py)  # X and Y are equal

    def test_biased_z_unity_is_symmetric(self):
        """bias=1 should give same result as symmetric depolarizing."""
        m_biased = PauliNoiseModel.from_biased_z(p=0.06, bias=1.0)
        m_symm = PauliNoiseModel.from_symmetric_depolarizing(p=0.06)
        assert m_biased.data_noise.px == pytest.approx(m_symm.data_noise.px)
        assert m_biased.data_noise.pz == pytest.approx(m_symm.data_noise.pz)

    def test_si1000_scaling(self):
        """SI1000: all parameters must scale correctly from base p."""
        p = 0.01
        model = PauliNoiseModel.from_si1000(p)
        # p_1q = p/10 = 0.001
        assert model.data_noise.total_error() == pytest.approx(p / 10.0)
        # p_2q = p
        assert model.prob_2q_gate == pytest.approx(p)
        # p_meas = 2p
        assert model.meas_flip == pytest.approx(2.0 * p)
        # p_reset = 5p
        assert model.reset_flip == pytest.approx(5.0 * p)
        # p_idle = p/20
        assert model.idle_noise is not None
        assert model.idle_noise.total_error() == pytest.approx(p / 20.0)

    def test_sd6_scaling(self):
        """SD6: all parameters must scale correctly from base p."""
        p = 0.01
        model = PauliNoiseModel.from_sd6(p)
        # p_1q = 0.1 * p
        assert model.data_noise.total_error() == pytest.approx(0.1 * p)
        # p_2q = p
        assert model.prob_2q_gate == pytest.approx(p)
        # p_meas = p
        assert model.meas_flip == pytest.approx(p)
        # p_reset = 0.2 * p
        assert model.reset_flip == pytest.approx(0.2 * p)
        # p_idle = 0.02 * p
        assert model.idle_noise is not None
        assert model.idle_noise.total_error() == pytest.approx(0.02 * p)

    def test_zero_noise(self):
        """p=0 should produce a completely clean model."""
        model = PauliNoiseModel.from_symmetric_depolarizing(p=0.0)
        assert model.data_noise.total_error() == 0.0
        assert model.meas_flip == 0.0


# =========================================================================
# Canonicalization & hashing tests
# =========================================================================

class TestCanonicalization:

    def test_canonical_hash_stable(self):
        """Same inputs must always produce the same hash."""
        m1 = PauliNoiseModel.from_si1000(0.01)
        m2 = PauliNoiseModel.from_si1000(0.01)
        assert m1.canonical_hash() == m2.canonical_hash()

    def test_canonical_hash_differs(self):
        """Different inputs must produce different hashes."""
        m1 = PauliNoiseModel.from_si1000(0.01)
        m2 = PauliNoiseModel.from_si1000(0.02)
        assert m1.canonical_hash() != m2.canonical_hash()

    def test_canonical_dict_sorted(self):
        """canonical_dict keys must be sorted for stable serialization."""
        model = PauliNoiseModel.from_si1000(0.01)
        d = model.canonical_dict()
        keys = list(d.keys())
        assert keys == sorted(keys)

    def test_canonical_hash_different_presets(self):
        """SD6 and SI1000 at same p must produce different hashes (different ratios)."""
        m_sd6 = PauliNoiseModel.from_sd6(0.01)
        m_si = PauliNoiseModel.from_si1000(0.01)
        assert m_sd6.canonical_hash() != m_si.canonical_hash()


# =========================================================================
# Model utilities tests
# =========================================================================

class TestModelUtilities:

    def test_scale_model(self):
        """scale(0.5) should halve all noise rates."""
        m = PauliNoiseModel.from_symmetric_depolarizing(p=0.06, meas_flip=0.02)
        ms = m.scale(0.5)
        assert ms.data_noise.total_error() == pytest.approx(0.03)
        assert ms.meas_flip == pytest.approx(0.01)

    def test_compose_models(self):
        """compose() should add error rates from two models."""
        m1 = PauliNoiseModel.from_symmetric_depolarizing(p=0.03)
        m2 = PauliNoiseModel.from_symmetric_depolarizing(p=0.03)
        mc = m1.compose(m2)
        assert mc.data_noise.total_error() == pytest.approx(0.06)

    def test_monotonicity(self):
        """Higher p must always produce higher error rates."""
        for preset in [PauliNoiseModel.from_symmetric_depolarizing,
                       PauliNoiseModel.from_sd6,
                       PauliNoiseModel.from_si1000]:
            m_lo = preset(0.001)
            m_hi = preset(0.01)
            assert m_hi.data_noise.total_error() > m_lo.data_noise.total_error()


# =========================================================================
# NoiseCompiler tests
# =========================================================================

class TestNoiseCompiler:

    def test_compile_phenomenological(self):
        """Model → Stim phenomenological params dict."""
        model = PauliNoiseModel.from_symmetric_depolarizing(p=0.03, meas_flip=0.01)
        params = compile_to_phenomenological(model)
        assert "after_clifford_depolarization" in params
        assert params["after_clifford_depolarization"] == pytest.approx(0.03)
        assert params["before_measure_flip_probability"] == pytest.approx(0.01)

    def test_compile_circuit_level(self):
        """Model → CircuitLevelParams."""
        model = PauliNoiseModel.from_si1000(0.01)
        clp = compile_to_circuit_level(model)
        assert clp.p1 == pytest.approx(0.001)    # p/10
        assert clp.p2 == pytest.approx(0.01)     # p
        assert clp.p_meas == pytest.approx(0.02)  # 2p
        assert clp.p_reset == pytest.approx(0.05) # 5p

    def test_compile_biased_channel(self):
        """Model → (px, py, pz) tuple preserving bias."""
        model = PauliNoiseModel.from_biased_z(p=0.1, bias=10.0)
        px, py, pz = compile_to_pauli_channel_1(model)
        assert px == pytest.approx(py)
        assert pz / px == pytest.approx(10.0)
        assert px + py + pz == pytest.approx(0.1)

    def test_model_from_preset(self):
        """model_from_preset should return correct model types."""
        m1 = model_from_preset("baseline_symmetric", 0.01)
        assert m1.data_noise.total_error() == pytest.approx(0.01)

        m2 = model_from_preset("biased_z", 0.1, bias=50)
        assert m2.data_noise.pz / m2.data_noise.px == pytest.approx(50.0)

        m3 = model_from_preset("sd6_like", 0.01)
        assert m3.prob_2q_gate == pytest.approx(0.01)

        m4 = model_from_preset("si1000_like", 0.01)
        assert m4.prob_2q_gate == pytest.approx(0.01)

    def test_model_from_preset_unknown_raises(self):
        """Unknown preset name must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown"):
            model_from_preset("nonexistent", 0.01)


# =========================================================================
# Serialization tests (to_dict / from_dict)
# =========================================================================

class TestSerialization:

    def test_channel_roundtrip(self):
        """PauliChannel to_dict → from_dict roundtrip."""
        ch = PauliChannel(px=0.1, py=0.2, pz=0.3)
        d = ch.to_dict()
        ch2 = PauliChannel.from_dict(d)
        assert ch2.px == pytest.approx(ch.px)
        assert ch2.py == pytest.approx(ch.py)
        assert ch2.pz == pytest.approx(ch.pz)

    def test_model_roundtrip(self):
        """PauliNoiseModel to_dict → from_dict roundtrip preserves all fields."""
        model = PauliNoiseModel.from_si1000(0.01)
        d = model.to_dict()
        model2 = PauliNoiseModel.from_dict(d)
        assert model2.data_noise.total_error() == pytest.approx(model.data_noise.total_error())
        assert model2.meas_flip == pytest.approx(model.meas_flip)
        assert model2.reset_flip == pytest.approx(model.reset_flip)
        assert model2.prob_2q_gate == pytest.approx(model.prob_2q_gate)
        assert model2.idle_noise is not None
        assert model2.idle_noise.total_error() == pytest.approx(model.idle_noise.total_error())
        assert model2.name == model.name
        assert model2.schema_version == model.schema_version

    def test_model_roundtrip_hash_stable(self):
        """Roundtripped model must produce the same canonical hash."""
        model = PauliNoiseModel.from_sd6(0.005)
        d = model.to_dict()
        model2 = PauliNoiseModel.from_dict(d)
        assert model.canonical_hash() == model2.canonical_hash()


# =========================================================================
# Metadata tests
# =========================================================================

class TestMetadata:

    def test_preset_names(self):
        """Each preset must set the correct name."""
        assert PauliNoiseModel.from_symmetric_depolarizing(0.01).name == "baseline_symmetric"
        assert PauliNoiseModel.from_biased_z(0.01).name == "biased_z"
        assert PauliNoiseModel.from_sd6(0.01).name == "sd6_like"
        assert PauliNoiseModel.from_si1000(0.01).name == "si1000_like"

    def test_schema_version_default(self):
        """All models should default to schema_version=1."""
        for preset in [PauliNoiseModel.from_symmetric_depolarizing,
                       PauliNoiseModel.from_sd6,
                       PauliNoiseModel.from_si1000]:
            assert preset(0.01).schema_version == 1
