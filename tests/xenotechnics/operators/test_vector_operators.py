"""
Tests for vector-based operators.

Tests for xenotechnics/operators/vector_operators.py
"""

from __future__ import annotations

import numpy as np
import pytest

from xenotechnics.common import String
from xenotechnics.operators import (
    L1DifferenceOperator,
    L1ScoreOperator,
    L2DifferenceOperator,
    L2ScoreOperator,
    L2SquaredDifferenceOperator,
    L2SquaredScoreOperator,
    LinfDifferenceOperator,
    LinfScoreOperator,
    MeanScoreOperator,
)
from xenotechnics.systems.vector_system import VectorSystemCompliance


class MockVectorSystem:
    """Mock system for testing."""

    def __init__(self, n: int = 3):
        self.n = n

    def __len__(self):
        return self.n


def make_compliance(values: list[float]) -> VectorSystemCompliance:
    """Helper to create VectorSystemCompliance for testing."""
    system = MockVectorSystem(len(values))
    return VectorSystemCompliance(
        system=system, compliance_vector=np.array(values), string=String.empty()
    )


class TestL2ScoreOperator:
    """Test L2ScoreOperator class."""

    def test_basic_computation(self):
        """Test basic L2 norm computation."""
        operator = L2ScoreOperator()
        compliance = make_compliance([0.3, 0.4, 0.0])

        # L2 norm = sqrt(0.09 + 0.16 + 0) = sqrt(0.25) = 0.5
        # Normalized by sqrt(3) ≈ 1.732
        result = operator(compliance)
        expected = 0.5 / np.sqrt(3)
        assert result == pytest.approx(expected)

    def test_all_zeros(self):
        """Test with all zero compliances."""
        operator = L2ScoreOperator()
        compliance = make_compliance([0.0, 0.0, 0.0])
        assert operator(compliance) == 0.0

    def test_all_ones(self):
        """Test with all one compliances."""
        operator = L2ScoreOperator()
        compliance = make_compliance([1.0, 1.0, 1.0])
        # L2 norm = sqrt(3), normalized by sqrt(3) = 1.0
        assert operator(compliance) == pytest.approx(1.0)

    def test_single_value(self):
        """Test with single value."""
        operator = L2ScoreOperator()
        compliance = make_compliance([0.5])
        # L2 norm = 0.5, normalized by sqrt(1) = 0.5
        assert operator(compliance) == pytest.approx(0.5)


class TestL2SquaredScoreOperator:
    """Test L2SquaredScoreOperator class."""

    def test_basic_computation(self):
        """Test basic squared L2 norm computation."""
        operator = L2SquaredScoreOperator()
        compliance = make_compliance([0.3, 0.4, 0.0])

        # Squared L2 = 0.09 + 0.16 + 0 = 0.25
        # Normalized by n=3
        result = operator(compliance)
        expected = 0.25 / 3
        assert result == pytest.approx(expected)

    def test_all_ones(self):
        """Test with all one compliances."""
        operator = L2SquaredScoreOperator()
        compliance = make_compliance([1.0, 1.0, 1.0])
        # Squared L2 = 3, normalized by 3 = 1.0
        assert operator(compliance) == pytest.approx(1.0)


class TestL1ScoreOperator:
    """Test L1ScoreOperator class."""

    def test_basic_computation(self):
        """Test basic L1 norm computation."""
        operator = L1ScoreOperator()
        compliance = make_compliance([0.2, 0.3, 0.5])

        # L1 norm = 0.2 + 0.3 + 0.5 = 1.0
        # Normalized by n=3
        result = operator(compliance)
        expected = 1.0 / 3
        assert result == pytest.approx(expected)

    def test_all_ones(self):
        """Test with all one compliances."""
        operator = L1ScoreOperator()
        compliance = make_compliance([1.0, 1.0, 1.0])
        # L1 norm = 3, normalized by 3 = 1.0
        assert operator(compliance) == pytest.approx(1.0)


class TestLinfScoreOperator:
    """Test LinfScoreOperator class."""

    def test_basic_computation(self):
        """Test basic L-infinity norm computation."""
        operator = LinfScoreOperator()
        compliance = make_compliance([0.2, 0.8, 0.5])

        # L-inf norm = max(|0.2|, |0.8|, |0.5|) = 0.8
        assert operator(compliance) == pytest.approx(0.8)

    def test_all_same(self):
        """Test with all same values."""
        operator = LinfScoreOperator()
        compliance = make_compliance([0.5, 0.5, 0.5])
        assert operator(compliance) == pytest.approx(0.5)


class TestMeanScoreOperator:
    """Test MeanScoreOperator class."""

    def test_basic_computation(self):
        """Test basic mean computation."""
        operator = MeanScoreOperator()
        compliance = make_compliance([0.2, 0.4, 0.6])

        # Mean = (0.2 + 0.4 + 0.6) / 3 = 0.4
        assert operator(compliance) == pytest.approx(0.4)

    def test_all_ones(self):
        """Test with all one compliances."""
        operator = MeanScoreOperator()
        compliance = make_compliance([1.0, 1.0, 1.0])
        assert operator(compliance) == pytest.approx(1.0)

    def test_all_zeros(self):
        """Test with all zero compliances."""
        operator = MeanScoreOperator()
        compliance = make_compliance([0.0, 0.0, 0.0])
        assert operator(compliance) == pytest.approx(0.0)


class TestL2DifferenceOperator:
    """Test L2DifferenceOperator class."""

    def test_basic_computation(self):
        """Test basic L2 difference computation."""
        operator = L2DifferenceOperator()
        c1 = make_compliance([0.8, 0.6, 0.4])
        c2 = make_compliance([0.5, 0.5, 0.5])

        # Diff = [0.3, 0.1, -0.1]
        # L2 = sqrt(0.09 + 0.01 + 0.01) = sqrt(0.11)
        # Normalized by sqrt(3)
        diff = np.array([0.3, 0.1, -0.1])
        expected = np.linalg.norm(diff) / np.sqrt(3)
        assert operator(c1, c2) == pytest.approx(expected)

    def test_same_compliances(self):
        """Test with identical compliances."""
        operator = L2DifferenceOperator()
        c1 = make_compliance([0.5, 0.5, 0.5])
        c2 = make_compliance([0.5, 0.5, 0.5])
        assert operator(c1, c2) == pytest.approx(0.0)

    def test_opposite_compliances(self):
        """Test with opposite compliances."""
        operator = L2DifferenceOperator()
        c1 = make_compliance([1.0, 1.0, 1.0])
        c2 = make_compliance([0.0, 0.0, 0.0])

        # Diff = [1, 1, 1], L2 = sqrt(3), normalized by sqrt(3) = 1
        assert operator(c1, c2) == pytest.approx(1.0)


class TestL2SquaredDifferenceOperator:
    """Test L2SquaredDifferenceOperator class."""

    def test_basic_computation(self):
        """Test basic squared L2 difference computation."""
        operator = L2SquaredDifferenceOperator()
        c1 = make_compliance([0.8, 0.6, 0.4])
        c2 = make_compliance([0.5, 0.5, 0.5])

        # Diff = [0.3, 0.1, -0.1]
        # Squared L2 = 0.09 + 0.01 + 0.01 = 0.11
        # Normalized by n=3
        expected = 0.11 / 3
        assert operator(c1, c2) == pytest.approx(expected)


class TestL1DifferenceOperator:
    """Test L1DifferenceOperator class."""

    def test_basic_computation(self):
        """Test basic L1 difference computation."""
        operator = L1DifferenceOperator()
        c1 = make_compliance([0.8, 0.6, 0.4])
        c2 = make_compliance([0.5, 0.5, 0.5])

        # Diff = [0.3, 0.1, -0.1]
        # L1 = 0.3 + 0.1 + 0.1 = 0.5
        # Normalized by n=3
        expected = 0.5 / 3
        assert operator(c1, c2) == pytest.approx(expected)


class TestLinfDifferenceOperator:
    """Test LinfDifferenceOperator class."""

    def test_basic_computation(self):
        """Test basic L-infinity difference computation."""
        operator = LinfDifferenceOperator()
        c1 = make_compliance([0.8, 0.6, 0.4])
        c2 = make_compliance([0.5, 0.5, 0.5])

        # Diff = [0.3, 0.1, -0.1]
        # L-inf = max(|0.3|, |0.1|, |-0.1|) = 0.3
        assert operator(c1, c2) == pytest.approx(0.3)

    def test_same_compliances(self):
        """Test with identical compliances."""
        operator = LinfDifferenceOperator()
        c1 = make_compliance([0.5, 0.5, 0.5])
        c2 = make_compliance([0.5, 0.5, 0.5])
        assert operator(c1, c2) == pytest.approx(0.0)
