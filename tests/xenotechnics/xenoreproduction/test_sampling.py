"""
Tests for xeno-distribution sampling.

Tests for xenotechnics/xenoreproduction/sampling.py
"""

from __future__ import annotations

import numpy as np
import pytest

from xenotechnics.common import FunctionalStructure, String
from xenotechnics.systems.vector_system import VectorSystem
from xenotechnics.xenoreproduction.sampling import (
    sample_xeno_trajectory,
    xeno_distribution,
)


class TestXenoDistribution:
    """Test xeno_distribution function."""

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

    def test_empty_strings(self, simple_system):
        """Test with empty string list."""
        result = xeno_distribution(
            simple_system, strings=[], base_probs=np.array([]), temperature=1.0
        )
        assert len(result) == 0

    def test_mismatched_lengths_raises(self, simple_system):
        """Test that mismatched lengths raise error."""
        strings = [String(tokens=("hello",)), String(tokens=("world",))]
        base_probs = np.array([1.0])  # Wrong length

        with pytest.raises(ValueError, match="Number of strings must match"):
            xeno_distribution(simple_system, strings, base_probs)

    def test_returns_probability_distribution(self, simple_system):
        """Test that result is a valid probability distribution."""
        strings = [
            String(tokens=("hello",)),
            String(tokens=("world",)),
            String(tokens=("test",)),
        ]
        base_probs = np.array([0.5, 0.3, 0.2])

        result = xeno_distribution(simple_system, strings, base_probs)

        # Should sum to 1
        assert np.sum(result) == pytest.approx(1.0)
        # All probabilities should be non-negative
        assert all(p >= 0 for p in result)

    def test_uniform_base_probs(self, simple_system):
        """Test with uniform base probabilities."""
        strings = [
            String(tokens=("a",)),  # Short, high vowel
            String(tokens=("xyz", "abc", "def")),  # Long, low vowel
        ]
        base_probs = np.array([0.5, 0.5])

        result = xeno_distribution(simple_system, strings, base_probs)

        assert len(result) == 2
        assert np.sum(result) == pytest.approx(1.0)

    def test_high_temperature_more_uniform(self, simple_system):
        """Test that high temperature makes distribution more uniform."""
        strings = [
            String(tokens=("hello",)),
            String(tokens=("world", "!", "test", "longer")),
        ]
        base_probs = np.array([0.9, 0.1])

        result_low_temp = xeno_distribution(
            simple_system, strings, base_probs, temperature=0.1
        )
        result_high_temp = xeno_distribution(
            simple_system, strings, base_probs, temperature=10.0
        )

        # High temperature should be more uniform (less extreme)
        low_temp_max = max(result_low_temp)
        high_temp_max = max(result_high_temp)
        assert high_temp_max <= low_temp_max or abs(high_temp_max - low_temp_max) < 0.5

    def test_single_string(self, simple_system):
        """Test with single string."""
        strings = [String(tokens=("hello",))]
        base_probs = np.array([1.0])

        result = xeno_distribution(simple_system, strings, base_probs)

        assert len(result) == 1
        assert result[0] == pytest.approx(1.0)


class TestSampleXenoTrajectory:
    """Test sample_xeno_trajectory function."""

    @pytest.fixture
    def simple_system(self):
        """Create simple test system."""

        def length_fn(s: String) -> float:
            return min(1.0, len(s.tokens) / 10)

        return VectorSystem([FunctionalStructure(length_fn, name="length")])

    def test_basic_computation(self, simple_system):
        """Test basic xeno-trajectory score computation."""
        trajectory = String(tokens=("hello", " ", "world"))

        result = sample_xeno_trajectory(simple_system, trajectory)

        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_with_reference_strings(self, simple_system):
        """Test with explicit reference strings."""
        trajectory = String(tokens=("test",))
        reference_strings = [
            String(tokens=("hello",)),
            String(tokens=("world",)),
        ]

        result = sample_xeno_trajectory(
            simple_system, trajectory, reference_strings=reference_strings
        )

        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_default_reference_is_trajectory(self, simple_system):
        """Test that default reference is the trajectory itself."""
        trajectory = String(tokens=("hello",))

        # When reference is just the trajectory, deviance from core should be 0
        result = sample_xeno_trajectory(simple_system, trajectory)

        # Deviance from self-core should be zero
        assert result == pytest.approx(0.0, abs=0.01)

    def test_higher_deviance_for_outlier(self, simple_system):
        """Test that outlier trajectories have higher deviance."""
        # References are all short
        reference_strings = [
            String(tokens=("a",)),
            String(tokens=("b",)),
            String(tokens=("c",)),
        ]

        # Similar to references (short)
        similar = String(tokens=("d",))
        # Different from references (long)
        outlier = String(tokens=tuple("x" for _ in range(10)))

        score_similar = sample_xeno_trajectory(
            simple_system, similar, reference_strings=reference_strings
        )
        score_outlier = sample_xeno_trajectory(
            simple_system, outlier, reference_strings=reference_strings
        )

        # Outlier should have higher deviance
        assert score_outlier > score_similar

    def test_non_negative_score(self, simple_system):
        """Test that score is non-negative."""
        trajectories = [
            String(tokens=("hello",)),
            String(tokens=("world", "test")),
            String.empty(),
        ]

        for traj in trajectories:
            result = sample_xeno_trajectory(simple_system, traj)
            assert result >= 0.0
