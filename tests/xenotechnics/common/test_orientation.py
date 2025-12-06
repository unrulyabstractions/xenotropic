"""
Tests for Orientation class.

Tests for xenotechnics/common/orientation.py
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from xenotechnics.common import AbstractSystemCompliance, Orientation
from xenotechnics.common.operator import AbstractDifferenceOperator


class MockDifferenceOperator(AbstractDifferenceOperator):
    """Mock difference operator for testing."""

    def __init__(self, return_value: float = 0.5):
        self.return_value = return_value
        self.call_count = 0

    def __call__(
        self,
        compliance1: AbstractSystemCompliance,
        compliance2: AbstractSystemCompliance,
    ) -> float:
        self.call_count += 1
        return self.return_value


class TestOrientation:
    """Test Orientation class."""

    @pytest.fixture
    def mock_compliance_pair(self):
        """Create pair of mock compliances."""
        c1 = MagicMock(spec=AbstractSystemCompliance)
        c1.__len__ = MagicMock(return_value=3)

        c2 = MagicMock(spec=AbstractSystemCompliance)
        c2.__len__ = MagicMock(return_value=3)

        return c1, c2

    def test_init(self, mock_compliance_pair):
        """Test Orientation initialization."""
        c1, c2 = mock_compliance_pair
        diff_op = MockDifferenceOperator()

        orientation = Orientation(c1, c2, diff_op)

        assert orientation.compliance_left == c1
        assert orientation.compliance_right == c2
        assert orientation.difference_operator == diff_op

    def test_init_dimension_mismatch_raises(self):
        """Test initialization with mismatched dimensions raises error."""
        c1 = MagicMock(spec=AbstractSystemCompliance)
        c1.__len__ = MagicMock(return_value=3)

        c2 = MagicMock(spec=AbstractSystemCompliance)
        c2.__len__ = MagicMock(return_value=5)

        diff_op = MockDifferenceOperator()

        with pytest.raises(ValueError, match="same dimension"):
            Orientation(c1, c2, diff_op)

    def test_deviance(self, mock_compliance_pair):
        """Test deviance method calls difference operator."""
        c1, c2 = mock_compliance_pair
        diff_op = MockDifferenceOperator(return_value=0.75)

        orientation = Orientation(c1, c2, diff_op)
        deviance = orientation.deviance()

        assert deviance == 0.75
        assert diff_op.call_count == 1

    def test_deviance_returns_operator_result(self, mock_compliance_pair):
        """Test deviance returns exactly what operator returns."""
        c1, c2 = mock_compliance_pair

        for expected in [0.0, 0.25, 0.5, 0.75, 1.0]:
            diff_op = MockDifferenceOperator(return_value=expected)
            orientation = Orientation(c1, c2, diff_op)
            assert orientation.deviance() == expected

    def test_len(self, mock_compliance_pair):
        """Test __len__ returns compliance dimension."""
        c1, c2 = mock_compliance_pair
        diff_op = MockDifferenceOperator()

        orientation = Orientation(c1, c2, diff_op)

        assert len(orientation) == 3

    def test_len_with_different_dimensions(self):
        """Test __len__ with various dimensions."""
        for dim in [1, 5, 10, 100]:
            c1 = MagicMock(spec=AbstractSystemCompliance)
            c1.__len__ = MagicMock(return_value=dim)

            c2 = MagicMock(spec=AbstractSystemCompliance)
            c2.__len__ = MagicMock(return_value=dim)

            diff_op = MockDifferenceOperator()
            orientation = Orientation(c1, c2, diff_op)

            assert len(orientation) == dim

    def test_repr(self, mock_compliance_pair):
        """Test __repr__ method."""
        c1, c2 = mock_compliance_pair
        diff_op = MockDifferenceOperator()

        orientation = Orientation(c1, c2, diff_op)
        rep = repr(orientation)

        assert "Orientation" in rep
        assert "dim=3" in rep


class TestOrientationEdgeCases:
    """Test Orientation edge cases."""

    def test_zero_deviance(self):
        """Test when deviance is zero (identical compliances)."""
        c1 = MagicMock(spec=AbstractSystemCompliance)
        c1.__len__ = MagicMock(return_value=3)

        c2 = MagicMock(spec=AbstractSystemCompliance)
        c2.__len__ = MagicMock(return_value=3)

        diff_op = MockDifferenceOperator(return_value=0.0)
        orientation = Orientation(c1, c2, diff_op)

        assert orientation.deviance() == 0.0

    def test_max_deviance(self):
        """Test when deviance is maximal."""
        c1 = MagicMock(spec=AbstractSystemCompliance)
        c1.__len__ = MagicMock(return_value=3)

        c2 = MagicMock(spec=AbstractSystemCompliance)
        c2.__len__ = MagicMock(return_value=3)

        diff_op = MockDifferenceOperator(return_value=1.0)
        orientation = Orientation(c1, c2, diff_op)

        assert orientation.deviance() == 1.0

    def test_single_dimension(self):
        """Test with single dimension."""
        c1 = MagicMock(spec=AbstractSystemCompliance)
        c1.__len__ = MagicMock(return_value=1)

        c2 = MagicMock(spec=AbstractSystemCompliance)
        c2.__len__ = MagicMock(return_value=1)

        diff_op = MockDifferenceOperator(return_value=0.5)
        orientation = Orientation(c1, c2, diff_op)

        assert len(orientation) == 1
        assert orientation.deviance() == 0.5

    def test_multiple_deviance_calls(self):
        """Test deviance can be called multiple times."""
        c1 = MagicMock(spec=AbstractSystemCompliance)
        c1.__len__ = MagicMock(return_value=3)

        c2 = MagicMock(spec=AbstractSystemCompliance)
        c2.__len__ = MagicMock(return_value=3)

        diff_op = MockDifferenceOperator(return_value=0.5)
        orientation = Orientation(c1, c2, diff_op)

        # Call multiple times
        for _ in range(5):
            assert orientation.deviance() == 0.5

        assert diff_op.call_count == 5
