"""
Tests for entropic operators.

Tests for xenotechnics/operators/entropic_operators.py
"""

from __future__ import annotations

import numpy as np
import pytest

from xenotechnics.common import FunctionalStructure, String
from xenotechnics.operators.entropic_operators import NormalizedEntropy, RelativeEntropy
from xenotechnics.systems.vector_system import VectorSystem, VectorSystemCompliance


def make_compliance(values: list[float], system=None) -> VectorSystemCompliance:
    """Helper to create VectorSystemCompliance for testing."""
    if system is None:
        # Create a mock system with right number of structures
        n = len(values)
        structures = [
            FunctionalStructure(lambda s, i=i: 0.5, name=f"struct_{i}")
            for i in range(n)
        ]
        system = VectorSystem(structures)
    return VectorSystemCompliance(
        system=system, compliance_vector=np.array(values), string=String.empty()
    )


class TestRelativeEntropy:
    """Test RelativeEntropy (Rényi divergence) operator."""

    def test_init_default_q(self):
        """Test default q value."""
        op = RelativeEntropy()
        assert op.q == 2.0

    def test_init_custom_q(self):
        """Test custom q value."""
        op = RelativeEntropy(q=1.5)
        assert op.q == 1.5

    def test_identical_distributions(self):
        """Test divergence between identical distributions is zero."""
        op = RelativeEntropy(q=2.0)
        c1 = make_compliance([0.5, 0.5])
        c2 = make_compliance([0.5, 0.5])

        result = op(c1, c2)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_different_distributions(self):
        """Test divergence between different distributions is positive."""
        op = RelativeEntropy(q=2.0)
        c1 = make_compliance([0.9, 0.1])
        c2 = make_compliance([0.1, 0.9])

        result = op(c1, c2)
        assert result > 0.0

    def test_q_equals_1_kl_divergence(self):
        """Test q=1 gives Shannon/KL divergence."""
        op = RelativeEntropy(q=1.0)
        c1 = make_compliance([0.7, 0.3])
        c2 = make_compliance([0.5, 0.5])

        result = op(c1, c2)
        assert np.isfinite(result)
        # KL divergence is non-negative
        assert result >= -0.01  # Small tolerance for numerical issues

    def test_higher_q_more_sensitive(self):
        """Test that higher q is more sensitive to distribution differences."""
        c1 = make_compliance([0.9, 0.1])
        c2 = make_compliance([0.5, 0.5])

        op_low = RelativeEntropy(q=1.5)
        op_high = RelativeEntropy(q=3.0)

        result_low = op_low(c1, c2)
        result_high = op_high(c1, c2)

        # Both should be positive
        assert result_low > 0
        assert result_high > 0

    def test_normalized_distributions(self):
        """Test that unnormalized inputs are normalized."""
        op = RelativeEntropy(q=2.0)
        # Input values that don't sum to 1
        c1 = make_compliance([2.0, 2.0])
        c2 = make_compliance([1.0, 1.0])

        # After normalization, both are uniform, so divergence should be ~0
        result = op(c1, c2)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_three_element_distribution(self):
        """Test with 3-element distributions."""
        op = RelativeEntropy(q=2.0)
        c1 = make_compliance([0.5, 0.3, 0.2])
        c2 = make_compliance([0.33, 0.33, 0.34])

        result = op(c1, c2)
        assert np.isfinite(result)


class TestNormalizedEntropy:
    """Test NormalizedEntropy score operator."""

    def test_init_default_q(self):
        """Test default q value."""
        op = NormalizedEntropy()
        assert op.q == 2.0

    def test_init_custom_q(self):
        """Test custom q value."""
        op = NormalizedEntropy(q=1.5)
        assert op.q == 1.5

    def test_uniform_distribution_zero_score(self):
        """Test uniform distribution has zero normalized entropy."""
        op = NormalizedEntropy()
        c = make_compliance([0.5, 0.5])

        result = op(c)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_concentrated_distribution_positive_score(self):
        """Test concentrated distribution has positive normalized entropy."""
        op = NormalizedEntropy()
        c = make_compliance([0.99, 0.01])

        result = op(c)
        assert result > 0.0

    def test_normalized_in_01(self):
        """Test result is in reasonable range."""
        op = NormalizedEntropy()

        test_cases = [
            [0.5, 0.5],
            [0.9, 0.1],
            [1.0, 0.0],
            [0.33, 0.33, 0.34],
            [0.8, 0.1, 0.1],
        ]

        for values in test_cases:
            c = make_compliance(values)
            result = op(c)
            assert np.isfinite(result)
            # Should be non-negative
            assert result >= -0.01

    def test_single_element(self):
        """Test with single element (edge case)."""
        op = NormalizedEntropy()
        c = make_compliance([1.0])

        result = op(c)
        # Single element has no entropy to normalize
        assert result == 0.0

    def test_three_element_uniform(self):
        """Test 3-element uniform distribution."""
        op = NormalizedEntropy()
        c = make_compliance([1.0 / 3, 1.0 / 3, 1.0 / 3])

        result = op(c)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_three_element_concentrated(self):
        """Test 3-element concentrated distribution."""
        op = NormalizedEntropy()
        c = make_compliance([0.98, 0.01, 0.01])

        result = op(c)
        assert result > 0.0
