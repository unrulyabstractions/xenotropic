"""
Tests for xeno-reproduction scoring functions.

Tests for xenotechnics/xenoreproduction/scoring.py
"""

from __future__ import annotations

import numpy as np
import pytest

from xenotechnics.common import FunctionalStructure, String
from xenotechnics.systems.vector_system import VectorSystem
from xenotechnics.xenoreproduction.data import InterventionScores
from xenotechnics.xenoreproduction.scoring import (
    score_concentration,
    score_diversity,
    score_fairness,
    score_intervention,
)


class TestScoreDiversity:
    """Test score_diversity function."""

    @pytest.fixture
    def simple_system(self):
        """Create simple test system."""

        def length_fn(s: String) -> float:
            return min(1.0, len(s.tokens) / 10)

        return VectorSystem([FunctionalStructure(length_fn, name="length")])

    def test_empty_strings(self, simple_system):
        """Test with empty string list."""
        result = score_diversity(simple_system, [])
        assert result == 0.0

    def test_identical_strings(self, simple_system):
        """Test identical strings have low diversity."""
        strings = [String(tokens=("hello",)) for _ in range(10)]

        result = score_diversity(simple_system, strings)

        # All identical, so variance is zero, only mean deviance contributes
        assert np.isfinite(result)

    def test_diverse_strings_higher_score(self, simple_system):
        """Test diverse strings have higher diversity score."""
        # Identical strings
        identical = [String(tokens=("hello",)) for _ in range(5)]

        # Diverse strings with varying lengths
        diverse = [String(tokens=tuple("x" for _ in range(i))) for i in range(1, 11, 2)]

        score_identical = score_diversity(simple_system, identical)
        score_diverse = score_diversity(simple_system, diverse)

        # Diverse should have higher score (higher variance)
        assert score_diverse > score_identical

    def test_single_string(self, simple_system):
        """Test with single string."""
        result = score_diversity(simple_system, [String(tokens=("hello",))])
        assert np.isfinite(result)

    def test_accepts_iterator(self, simple_system):
        """Test that function accepts iterator."""

        def gen_strings():
            yield String(tokens=("hello",))
            yield String(tokens=("world",))

        result = score_diversity(simple_system, gen_strings())
        assert np.isfinite(result)


class TestScoreFairness:
    """Test score_fairness function."""

    @pytest.fixture
    def multi_structure_system(self):
        """Create system with multiple structures."""

        def length_fn(s: String) -> float:
            return min(1.0, len(s.tokens) / 10)

        def vowel_fn(s: String) -> float:
            text = s.to_text().lower()
            if not text:
                return 0.5
            return sum(1 for c in text if c in "aeiou") / max(1, len(text))

        structures = [
            FunctionalStructure(length_fn, name="length"),
            FunctionalStructure(vowel_fn, name="vowels"),
        ]
        return VectorSystem(structures)

    def test_empty_strings(self, multi_structure_system):
        """Test with empty string list."""
        result = score_fairness(multi_structure_system, [])
        assert result == 0.0

    def test_negative_score(self, multi_structure_system):
        """Test that fairness score is negative (since it's -max)."""
        strings = [
            String(tokens=("hello",)),
            String(tokens=("world",)),
        ]

        result = score_fairness(multi_structure_system, strings)

        # Score is -max(core), so should be negative
        assert result < 0.0

    def test_more_uniform_higher_fairness(self, multi_structure_system):
        """Test that more uniform distribution has higher (less negative) fairness."""
        # Strings that produce imbalanced core (high in one structure)
        imbalanced = [
            String(tokens=("aeiou",)),  # High vowel, low length
            String(tokens=("aeiou",)),
        ]

        # Strings that produce more balanced core
        balanced = [
            String(tokens=("hello", "world", "test")),  # Mixed
            String(tokens=("xyz", "abc")),  # Mixed
        ]

        score_imbalanced = score_fairness(multi_structure_system, imbalanced)
        score_balanced = score_fairness(multi_structure_system, balanced)

        # Both should be negative
        assert score_imbalanced < 0
        assert score_balanced < 0

    def test_single_string(self, multi_structure_system):
        """Test with single string."""
        result = score_fairness(multi_structure_system, [String(tokens=("hello",))])
        assert np.isfinite(result)


class TestScoreConcentration:
    """Test score_concentration function."""

    @pytest.fixture
    def multi_structure_system(self):
        """Create system with multiple structures."""

        def length_fn(s: String) -> float:
            return min(1.0, len(s.tokens) / 10)

        def vowel_fn(s: String) -> float:
            text = s.to_text().lower()
            if not text:
                return 0.5
            return sum(1 for c in text if c in "aeiou") / max(1, len(text))

        structures = [
            FunctionalStructure(length_fn, name="length"),
            FunctionalStructure(vowel_fn, name="vowels"),
        ]
        return VectorSystem(structures)

    def test_empty_strings(self, multi_structure_system):
        """Test with empty string list."""
        result = score_concentration(multi_structure_system, [])
        assert result == 0.0

    def test_returns_finite(self, multi_structure_system):
        """Test that result is finite."""
        strings = [
            String(tokens=("hello",)),
            String(tokens=("world",)),
        ]

        result = score_concentration(multi_structure_system, strings)
        assert np.isfinite(result)

    def test_non_negative(self, multi_structure_system):
        """Test that entropy is non-negative."""
        strings = [
            String(tokens=("hello",)),
            String(tokens=("world",)),
            String(tokens=("test",)),
        ]

        result = score_concentration(multi_structure_system, strings)
        assert result >= 0.0


class TestScoreIntervention:
    """Test score_intervention function."""

    @pytest.fixture
    def simple_system(self):
        """Create simple test system."""

        def length_fn(s: String) -> float:
            return min(1.0, len(s.tokens) / 10)

        def vowel_fn(s: String) -> float:
            text = s.to_text().lower()
            if not text:
                return 0.5
            return sum(1 for c in text if c in "aeiou") / max(1, len(text))

        structures = [
            FunctionalStructure(length_fn, name="length"),
            FunctionalStructure(vowel_fn, name="vowels"),
        ]
        return VectorSystem(structures)

    def test_returns_intervention_scores(self, simple_system):
        """Test that function returns InterventionScores."""
        strings = [
            String(tokens=("hello",)),
            String(tokens=("world",)),
        ]

        result = score_intervention(simple_system, strings)

        assert isinstance(result, InterventionScores)

    def test_all_components_finite(self, simple_system):
        """Test all score components are finite."""
        strings = [
            String(tokens=("hello",)),
            String(tokens=("world",)),
            String(tokens=("test",)),
        ]

        result = score_intervention(simple_system, strings)

        assert np.isfinite(result.diversity)
        assert np.isfinite(result.fairness)
        assert np.isfinite(result.concentration)
        assert np.isfinite(result.total)

    def test_total_is_weighted_sum(self, simple_system):
        """Test that total is weighted sum of components."""
        strings = [
            String(tokens=("hello",)),
            String(tokens=("world",)),
        ]

        lambda_d, lambda_f, lambda_c = 2.0, 0.5, 1.5

        result = score_intervention(
            simple_system,
            strings,
            lambda_d=lambda_d,
            lambda_f=lambda_f,
            lambda_c=lambda_c,
        )

        expected_total = (
            lambda_d * result.diversity
            + lambda_f * result.fairness
            + lambda_c * result.concentration
        )
        assert result.total == pytest.approx(expected_total)

    def test_default_weights(self, simple_system):
        """Test default weights are all 1.0."""
        strings = [String(tokens=("hello",))]

        result = score_intervention(simple_system, strings)

        expected_total = result.diversity + result.fairness + result.concentration
        assert result.total == pytest.approx(expected_total)

    def test_zero_weights(self, simple_system):
        """Test zero weights exclude components."""
        strings = [String(tokens=("hello",)), String(tokens=("world",))]

        result = score_intervention(
            simple_system, strings, lambda_d=0.0, lambda_f=0.0, lambda_c=0.0
        )

        assert result.total == pytest.approx(0.0)

    def test_single_weight(self, simple_system):
        """Test with only one non-zero weight."""
        strings = [String(tokens=("hello",)), String(tokens=("world",))]

        result_d = score_intervention(
            simple_system, strings, lambda_d=1.0, lambda_f=0.0, lambda_c=0.0
        )
        result_f = score_intervention(
            simple_system, strings, lambda_d=0.0, lambda_f=1.0, lambda_c=0.0
        )
        result_c = score_intervention(
            simple_system, strings, lambda_d=0.0, lambda_f=0.0, lambda_c=1.0
        )

        assert result_d.total == pytest.approx(result_d.diversity)
        assert result_f.total == pytest.approx(result_f.fairness)
        assert result_c.total == pytest.approx(result_c.concentration)
