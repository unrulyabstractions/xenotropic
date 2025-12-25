"""
Tests for operator abstract classes.

Tests for xenotechnics/common/operator.py
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from xenotechnics.common import AbstractSystemCompliance
from xenotechnics.common.operator import (
    AbstractDifferenceOperator,
    AbstractScoreOperator,
)


class ConcreteScoreOperator(AbstractScoreOperator):
    """Concrete implementation for testing."""

    def __call__(self, compliance: AbstractSystemCompliance) -> float:
        # Simple mock: return 0.5
        return 0.5


class ConcreteDifferenceOperator(AbstractDifferenceOperator):
    """Concrete implementation for testing."""

    def __call__(
        self,
        compliance1: AbstractSystemCompliance,
        compliance2: AbstractSystemCompliance,
    ) -> float:
        # Simple mock: return 0.25
        return 0.25


class TestAbstractScoreOperator:
    """Test AbstractScoreOperator class."""

    def test_call_returns_float(self):
        """Test __call__ returns a float."""
        operator = ConcreteScoreOperator()
        compliance = MagicMock(spec=AbstractSystemCompliance)

        result = operator(compliance)

        assert isinstance(result, float)
        assert result == 0.5

    def test_operator_is_callable(self):
        """Test operator is callable."""
        operator = ConcreteScoreOperator()
        assert callable(operator)

    def test_abstract_method_enforcement(self):
        """Test that abstract method must be implemented."""
        # Cannot instantiate abstract class directly
        with pytest.raises(TypeError):
            AbstractScoreOperator()


class TestAbstractDifferenceOperator:
    """Test AbstractDifferenceOperator class."""

    def test_call_returns_float(self):
        """Test __call__ returns a float."""
        operator = ConcreteDifferenceOperator()
        compliance1 = MagicMock(spec=AbstractSystemCompliance)
        compliance2 = MagicMock(spec=AbstractSystemCompliance)

        result = operator(compliance1, compliance2)

        assert isinstance(result, float)
        assert result == 0.25

    def test_operator_is_callable(self):
        """Test operator is callable."""
        operator = ConcreteDifferenceOperator()
        assert callable(operator)

    def test_abstract_method_enforcement(self):
        """Test that abstract method must be implemented."""
        # Cannot instantiate abstract class directly
        with pytest.raises(TypeError):
            AbstractDifferenceOperator()

    def test_takes_two_compliances(self):
        """Test operator takes two compliance arguments."""
        operator = ConcreteDifferenceOperator()
        c1 = MagicMock(spec=AbstractSystemCompliance)
        c2 = MagicMock(spec=AbstractSystemCompliance)

        # Should work with two compliances
        result = operator(c1, c2)
        assert result == 0.25


class TestOperatorComposition:
    """Test operators can be composed."""

    def test_score_operator_chaining(self):
        """Test score operators can be used in sequence."""

        class DoubleScoreOperator(AbstractScoreOperator):
            def __init__(self, base_operator: AbstractScoreOperator):
                self.base = base_operator

            def __call__(self, compliance: AbstractSystemCompliance) -> float:
                return min(1.0, self.base(compliance) * 2)

        base = ConcreteScoreOperator()  # Returns 0.5
        doubled = DoubleScoreOperator(base)

        compliance = MagicMock(spec=AbstractSystemCompliance)
        assert doubled(compliance) == 1.0  # min(1.0, 0.5 * 2)

    def test_difference_with_score_operators(self):
        """Test difference operator working with score values."""

        class ScoreBasedDifference(AbstractDifferenceOperator):
            def __init__(self, score_op: AbstractScoreOperator):
                self.score_op = score_op

            def __call__(
                self, c1: AbstractSystemCompliance, c2: AbstractSystemCompliance
            ) -> float:
                s1 = self.score_op(c1)
                s2 = self.score_op(c2)
                return abs(s1 - s2)

        score_op = ConcreteScoreOperator()
        diff_op = ScoreBasedDifference(score_op)

        c1 = MagicMock(spec=AbstractSystemCompliance)
        c2 = MagicMock(spec=AbstractSystemCompliance)

        # Both return 0.5, so difference is 0
        assert diff_op(c1, c2) == 0.0
