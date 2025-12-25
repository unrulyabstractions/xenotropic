"""
Tests for judge-based systems.

Tests for xenotechnics/systems/judge_vector_system.py,
xenotechnics/systems/judge_entropic_system.py,
xenotechnics/systems/judge_generalized_system.py
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from xenotechnics.common import String


class TestJudgeVectorSystem:
    """Test JudgeVectorSystem class."""

    def test_init_requires_questions(self):
        """Test that empty questions raises error."""
        from xenotechnics.systems.judge_vector_system import JudgeVectorSystem

        with pytest.raises(ValueError, match="requires at least one question"):
            JudgeVectorSystem(questions=[], model=MagicMock())

    def test_init_requires_model_or_model_name(self):
        """Test that missing model and model_name raises error."""
        from xenotechnics.systems.judge_vector_system import JudgeVectorSystem

        with pytest.raises(ValueError, match="Must provide either model or model_name"):
            JudgeVectorSystem(questions=["Test?"])

    @patch("xenotechnics.systems.judge_vector_system.JudgeStructure")
    def test_init_with_model(self, mock_judge_cls):
        """Test initialization with pre-loaded model."""
        from xenotechnics.systems.judge_vector_system import JudgeVectorSystem

        mock_model = MagicMock()
        mock_judge = MagicMock()
        mock_judge.compliance.return_value = 0.5
        mock_judge_cls.return_value = mock_judge

        questions = ["Is this good?", "Is this valid?"]
        system = JudgeVectorSystem(questions=questions, model=mock_model)

        assert len(system.questions) == 2
        assert len(system.structures) == 2
        assert mock_judge_cls.call_count == 2

    @patch("xenotechnics.systems.judge_vector_system.JudgeStructure")
    def test_init_with_model_name(self, mock_judge_cls):
        """Test initialization with model name."""
        from xenotechnics.systems.judge_vector_system import JudgeVectorSystem

        mock_judge = MagicMock()
        mock_judge.compliance.return_value = 0.5
        mock_judge_cls.return_value = mock_judge

        system = JudgeVectorSystem(questions=["Question 1?"], model_name="test-model")

        assert len(system.questions) == 1

    @patch("xenotechnics.systems.judge_vector_system.JudgeStructure")
    def test_repr(self, mock_judge_cls):
        """Test string representation."""
        from xenotechnics.systems.judge_vector_system import JudgeVectorSystem

        mock_judge = MagicMock()
        mock_judge_cls.return_value = mock_judge

        system = JudgeVectorSystem(questions=["Q1?", "Q2?", "Q3?"], model=MagicMock())

        assert "3 judges" in repr(system)


class TestJudgeEntropicSystem:
    """Test JudgeEntropicSystem class."""

    @patch("xenotechnics.systems.judge_vector_system.JudgeStructure")
    def test_init_excess_mode(self, mock_judge_cls):
        """Test initialization in excess mode."""
        from xenotechnics.systems.judge_entropic_system import JudgeEntropicSystem

        mock_judge = MagicMock()
        mock_judge_cls.return_value = mock_judge

        system = JudgeEntropicSystem(
            questions=["Test?"],
            model=MagicMock(),
            q=2.0,
            mode="excess",
        )

        assert system.mode == "excess"
        assert system.q == 2.0

    @patch("xenotechnics.systems.judge_vector_system.JudgeStructure")
    def test_init_deficit_mode(self, mock_judge_cls):
        """Test initialization in deficit mode."""
        from xenotechnics.systems.judge_entropic_system import JudgeEntropicSystem

        mock_judge = MagicMock()
        mock_judge_cls.return_value = mock_judge

        system = JudgeEntropicSystem(
            questions=["Test?"],
            model=MagicMock(),
            q=2.0,
            mode="deficit",
        )

        assert system.mode == "deficit"

    @patch("xenotechnics.systems.judge_vector_system.JudgeStructure")
    def test_init_invalid_mode_raises(self, mock_judge_cls):
        """Test that invalid mode raises error."""
        from xenotechnics.systems.judge_entropic_system import JudgeEntropicSystem

        mock_judge = MagicMock()
        mock_judge_cls.return_value = mock_judge

        with pytest.raises(ValueError, match="Invalid mode"):
            JudgeEntropicSystem(
                questions=["Test?"],
                model=MagicMock(),
                mode="invalid",
            )

    @patch("xenotechnics.systems.judge_vector_system.JudgeStructure")
    def test_repr_excess(self, mock_judge_cls):
        """Test repr for excess mode."""
        from xenotechnics.systems.judge_entropic_system import JudgeEntropicSystem

        mock_judge = MagicMock()
        mock_judge_cls.return_value = mock_judge

        system = JudgeEntropicSystem(
            questions=["Q1?", "Q2?"],
            model=MagicMock(),
            q=3.0,
            mode="excess",
        )

        rep = repr(system)
        assert "2 judges" in rep
        assert "q=3.0" in rep
        assert "mode=excess" in rep


class TestExcessDifferenceOperator:
    """Test ExcessDifferenceOperator."""

    def test_call(self):
        """Test excess deviance computation."""
        from xenotechnics.systems.judge_entropic_system import (
            ExcessDifferenceOperator,
        )

        # Create mock compliances (no spec to allow setting to_array)
        compliance1 = MagicMock()
        compliance1.to_array.return_value = np.array([0.8, 0.2])
        compliance1.__len__ = MagicMock(return_value=2)

        compliance2 = MagicMock()
        compliance2.to_array.return_value = np.array([0.5, 0.5])
        compliance2.__len__ = MagicMock(return_value=2)

        op = ExcessDifferenceOperator(q=2.0)

        # The operator uses RelativeEntropy which expects to_array()
        result = op(compliance1, compliance2)

        assert isinstance(result, float)
        assert np.isfinite(result)


class TestDeficitDifferenceOperator:
    """Test DeficitDifferenceOperator."""

    def test_call(self):
        """Test deficit deviance computation."""
        from xenotechnics.systems.judge_entropic_system import (
            DeficitDifferenceOperator,
        )

        # Create mock compliances (no spec to allow setting to_array)
        compliance1 = MagicMock()
        compliance1.to_array.return_value = np.array([0.5, 0.5])
        compliance1.__len__ = MagicMock(return_value=2)

        compliance2 = MagicMock()
        compliance2.to_array.return_value = np.array([0.8, 0.2])
        compliance2.__len__ = MagicMock(return_value=2)

        op = DeficitDifferenceOperator(q=2.0)

        # Arguments should be flipped internally
        result = op(compliance1, compliance2)

        assert isinstance(result, float)
        assert np.isfinite(result)


class TestJudgeGeneralizedSystem:
    """Test JudgeGeneralizedSystem class."""

    @patch("xenotechnics.systems.judge_vector_system.JudgeStructure")
    def test_init(self, mock_judge_cls):
        """Test initialization."""
        from xenotechnics.systems.judge_generalized_system import (
            JudgeGeneralizedSystem,
        )

        mock_judge = MagicMock()
        mock_judge_cls.return_value = mock_judge

        system = JudgeGeneralizedSystem(
            questions=["Test?"],
            model=MagicMock(),
            q=2.0,
            r=1.5,
        )

        assert system.q == 2.0
        assert system.r == 1.5

    @patch("xenotechnics.systems.judge_vector_system.JudgeStructure")
    def test_compute_core_empty_raises(self, mock_judge_cls):
        """Test that empty trajectories raises error."""
        from xenotechnics.systems.judge_generalized_system import (
            JudgeGeneralizedSystem,
        )

        mock_judge = MagicMock()
        mock_judge_cls.return_value = mock_judge

        system = JudgeGeneralizedSystem(
            questions=["Test?"],
            model=MagicMock(),
        )

        with pytest.raises(ValueError, match="empty trajectory"):
            system.compute_core([], np.array([]))

    @patch("xenotechnics.systems.judge_vector_system.JudgeStructure")
    def test_compute_core_mismatched_lengths_raises(self, mock_judge_cls):
        """Test that mismatched lengths raises error."""
        from xenotechnics.systems.judge_generalized_system import (
            JudgeGeneralizedSystem,
        )

        mock_judge = MagicMock()
        mock_judge_cls.return_value = mock_judge

        system = JudgeGeneralizedSystem(
            questions=["Test?"],
            model=MagicMock(),
        )

        trajectories = [String(tokens=("a",)), String(tokens=("b",))]
        probs = np.array([1.0])  # Wrong length

        with pytest.raises(ValueError, match="must match"):
            system.compute_core(trajectories, probs)

    @patch("xenotechnics.systems.judge_vector_system.JudgeStructure")
    def test_compute_core_basic(self, mock_judge_cls):
        """Test basic core computation."""
        from xenotechnics.systems.judge_generalized_system import (
            JudgeGeneralizedSystem,
        )
        from xenotechnics.systems.vector_system import VectorSystemCompliance

        mock_judge = MagicMock()
        mock_judge.compliance.side_effect = [0.6, 0.4]
        mock_judge_cls.return_value = mock_judge

        system = JudgeGeneralizedSystem(
            questions=["Test?"],
            model=MagicMock(),
            q=1.0,
            r=1.0,
        )

        trajectories = [String(tokens=("a",)), String(tokens=("b",))]
        probs = np.array([0.5, 0.5])

        result = system.compute_core(trajectories, probs)

        assert isinstance(result, VectorSystemCompliance)
        assert result.string is None  # Core has no associated string

    @patch("xenotechnics.systems.judge_vector_system.JudgeStructure")
    def test_compute_core_q_zero(self, mock_judge_cls):
        """Test core computation with q=0 (geometric mean)."""
        from xenotechnics.systems.judge_generalized_system import (
            JudgeGeneralizedSystem,
        )

        mock_judge = MagicMock()
        mock_judge.compliance.side_effect = [0.6, 0.4]
        mock_judge_cls.return_value = mock_judge

        system = JudgeGeneralizedSystem(
            questions=["Test?"],
            model=MagicMock(),
            q=0.0,
            r=1.0,
        )

        trajectories = [String(tokens=("a",)), String(tokens=("b",))]
        probs = np.array([0.5, 0.5])

        result = system.compute_core(trajectories, probs)

        # Should compute geometric mean
        core = result.to_array()
        assert np.isfinite(core[0])

    @patch("xenotechnics.systems.judge_vector_system.JudgeStructure")
    def test_repr(self, mock_judge_cls):
        """Test string representation."""
        from xenotechnics.systems.judge_generalized_system import (
            JudgeGeneralizedSystem,
        )

        mock_judge = MagicMock()
        mock_judge_cls.return_value = mock_judge

        system = JudgeGeneralizedSystem(
            questions=["Q1?", "Q2?"],
            model=MagicMock(),
            q=2.0,
            r=0.5,
        )

        rep = repr(system)
        assert "2 judges" in rep
        assert "q=2.0" in rep
        assert "r=0.5" in rep
