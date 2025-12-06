"""
Tests for singleton system.

Tests for xenotechnics/systems/singleton_system.py
"""

from __future__ import annotations

import numpy as np
import pytest

from xenotechnics.common import String
from xenotechnics.operators import L2DifferenceOperator, L2ScoreOperator
from xenotechnics.systems.singleton_system import SingletonSystem


class TestSingletonSystem:
    """Test SingletonSystem class."""

    def test_init_basic(self):
        """Test basic initialization."""

        def length_fn(s: String) -> float:
            return min(1.0, len(s.tokens) / 10)

        system = SingletonSystem(
            compliance_fn=length_fn,
            score_operator=L2ScoreOperator(),
            difference_operator=L2DifferenceOperator(),
            name="length",
            description="Length-based compliance",
        )

        assert len(system) == 1
        assert system.structure.name == "length"

    def test_init_default_name(self):
        """Test initialization with default name."""

        def dummy_fn(s: String) -> float:
            return 0.5

        system = SingletonSystem(
            compliance_fn=dummy_fn,
            score_operator=L2ScoreOperator(),
            difference_operator=L2DifferenceOperator(),
        )

        assert system.structure.name == "singleton"

    def test_compliance_returns_single_value(self):
        """Test compliance returns single-element vector."""

        def length_fn(s: String) -> float:
            return min(1.0, len(s.tokens) / 10)

        system = SingletonSystem(
            compliance_fn=length_fn,
            score_operator=L2ScoreOperator(),
            difference_operator=L2DifferenceOperator(),
        )

        string = String(tokens=("hello", " ", "world"))
        compliance = system.compliance(string)

        assert len(compliance.to_array()) == 1
        assert compliance.to_array()[0] == pytest.approx(0.3)  # 3/10

    def test_compliance_empty_string(self):
        """Test compliance with empty string."""

        def length_fn(s: String) -> float:
            return min(1.0, len(s.tokens) / 10)

        system = SingletonSystem(
            compliance_fn=length_fn,
            score_operator=L2ScoreOperator(),
            difference_operator=L2DifferenceOperator(),
        )

        string = String.empty()
        compliance = system.compliance(string)

        assert compliance.to_array()[0] == 0.0

    def test_structure_names(self):
        """Test structure_names returns list with single name."""

        def dummy_fn(s: String) -> float:
            return 0.5

        system = SingletonSystem(
            compliance_fn=dummy_fn,
            score_operator=L2ScoreOperator(),
            difference_operator=L2DifferenceOperator(),
            name="test_struct",
        )

        names = system.structure_names()
        assert names == ["test_struct"]

    def test_len(self):
        """Test __len__ returns 1."""

        def dummy_fn(s: String) -> float:
            return 0.5

        system = SingletonSystem(
            compliance_fn=dummy_fn,
            score_operator=L2ScoreOperator(),
            difference_operator=L2DifferenceOperator(),
        )

        assert len(system) == 1

    def test_compute_core(self):
        """Test compute_core with trajectories."""

        def length_fn(s: String) -> float:
            return min(1.0, len(s.tokens) / 10)

        system = SingletonSystem(
            compliance_fn=length_fn,
            score_operator=L2ScoreOperator(),
            difference_operator=L2DifferenceOperator(),
        )

        trajectories = [
            String(tokens=("a",)),  # 0.1
            String(tokens=("a", "b", "c")),  # 0.3
            String(tokens=("a", "b", "c", "d", "e")),  # 0.5
        ]
        probs = np.array([0.5, 0.3, 0.2])

        core = system.compute_core(trajectories, probs)

        # Weighted average: 0.5*0.1 + 0.3*0.3 + 0.2*0.5 = 0.05 + 0.09 + 0.1 = 0.24
        assert core.to_array()[0] == pytest.approx(0.24)

    def test_repr(self):
        """Test string representation."""

        def dummy_fn(s: String) -> float:
            return 0.5

        system = SingletonSystem(
            compliance_fn=dummy_fn,
            score_operator=L2ScoreOperator(),
            difference_operator=L2DifferenceOperator(),
            name="my_system",
        )

        repr_str = repr(system)
        assert "SingletonSystem" in repr_str
        assert "my_system" in repr_str

    def test_score_operator_property(self):
        """Test score_operator property."""

        def dummy_fn(s: String) -> float:
            return 0.5

        op = L2ScoreOperator()
        system = SingletonSystem(
            compliance_fn=dummy_fn,
            score_operator=op,
            difference_operator=L2DifferenceOperator(),
        )

        assert system.score_operator is op

    def test_difference_operator_property(self):
        """Test difference_operator property."""

        def dummy_fn(s: String) -> float:
            return 0.5

        op = L2DifferenceOperator()
        system = SingletonSystem(
            compliance_fn=dummy_fn,
            score_operator=L2ScoreOperator(),
            difference_operator=op,
        )

        assert system.difference_operator is op

    def test_vowel_ratio_compliance(self):
        """Test with vowel ratio compliance function."""

        def vowel_fn(s: String) -> float:
            text = s.to_text().lower()
            if not text:
                return 0.5
            return sum(1 for c in text if c in "aeiou") / max(1, len(text))

        system = SingletonSystem(
            compliance_fn=vowel_fn,
            score_operator=L2ScoreOperator(),
            difference_operator=L2DifferenceOperator(),
            name="vowels",
        )

        # "aeiou" is all vowels
        all_vowels = String(tokens=("aeiou",))
        assert system.compliance(all_vowels).to_array()[0] == pytest.approx(1.0)

        # "xyz" has no vowels
        no_vowels = String(tokens=("xyz",))
        assert system.compliance(no_vowels).to_array()[0] == pytest.approx(0.0)
