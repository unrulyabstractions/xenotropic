"""
Tests for homogenization metrics computation.

Tests for xenotechnics/xenoreproduction/metrics.py
"""

from __future__ import annotations

import numpy as np
import pytest

from xenotechnics.common import FunctionalStructure, String
from xenotechnics.systems.vector_system import VectorSystem
from xenotechnics.xenoreproduction.data import HomogenizationMetrics
from xenotechnics.xenoreproduction.metrics import compute_homogenization_metrics


class TestComputeHomogenizationMetrics:
    """Test compute_homogenization_metrics function."""

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

    def test_returns_homogenization_metrics(self, simple_system):
        """Test that function returns HomogenizationMetrics."""
        strings = [
            String(tokens=("hello",)),
            String(tokens=("world",)),
            String(tokens=("test",)),
        ]

        result = compute_homogenization_metrics(simple_system, strings)

        assert isinstance(result, HomogenizationMetrics)

    def test_metrics_are_finite(self, simple_system):
        """Test all metrics are finite."""
        strings = [
            String(tokens=("hello",)),
            String(tokens=("world",)),
            String(tokens=("test",)),
        ]

        result = compute_homogenization_metrics(simple_system, strings)

        assert np.isfinite(result.expected_deviance)
        assert np.isfinite(result.deviance_variance)
        assert np.isfinite(result.core_entropy)

    def test_empty_strings_returns_zero_metrics(self, simple_system):
        """Test empty string collection returns zero metrics."""
        result = compute_homogenization_metrics(simple_system, [])

        assert result.expected_deviance == 0.0
        assert result.deviance_variance == 0.0
        assert result.core_entropy == 0.0

    def test_identical_strings_low_variance(self, simple_system):
        """Test identical strings have low variance."""
        # All identical strings
        strings = [String(tokens=("hello",)) for _ in range(10)]

        result = compute_homogenization_metrics(simple_system, strings)

        # Variance should be zero for identical strings
        assert result.deviance_variance == pytest.approx(0.0, abs=0.01)

    def test_diverse_strings_higher_variance(self, simple_system):
        """Test diverse strings have higher variance."""
        # Diverse strings with different lengths and vowel ratios
        strings = [
            String(tokens=("a",)),  # Short, high vowel
            String(tokens=("xyz", "abc", "def", "ghi")),  # Long, low vowel
            String(tokens=("aeiou",)),  # Medium, all vowels
            String(tokens=("bcdfg",)),  # Medium, no vowels
            String(tokens=("hello", " ", "world", " ", "test")),  # Long, mixed
        ]

        result = compute_homogenization_metrics(simple_system, strings)

        # Variance should be non-zero for diverse strings
        assert result.deviance_variance > 0.0

    def test_single_string_zero_variance(self, simple_system):
        """Test single string has zero variance."""
        strings = [String(tokens=("hello",))]

        result = compute_homogenization_metrics(simple_system, strings)

        # Single sample has no variance
        assert result.deviance_variance == 0.0

    def test_expected_deviance_non_negative(self, simple_system):
        """Test expected deviance is non-negative."""
        strings = [
            String(tokens=("hello",)),
            String(tokens=("world",)),
            String(tokens=("test",)),
        ]

        result = compute_homogenization_metrics(simple_system, strings)

        assert result.expected_deviance >= 0.0

    def test_core_entropy_non_negative(self, simple_system):
        """Test core entropy is non-negative."""
        strings = [
            String(tokens=("hello",)),
            String(tokens=("world",)),
            String(tokens=("test",)),
        ]

        result = compute_homogenization_metrics(simple_system, strings)

        assert result.core_entropy >= 0.0

    def test_accepts_iterator(self, simple_system):
        """Test function accepts iterator (not just list)."""

        def string_generator():
            yield String(tokens=("hello",))
            yield String(tokens=("world",))
            yield String(tokens=("test",))

        result = compute_homogenization_metrics(simple_system, string_generator())

        assert isinstance(result, HomogenizationMetrics)
        assert np.isfinite(result.expected_deviance)


class TestHomogenizationTrends:
    """Test expected trends in homogenization metrics."""

    @pytest.fixture
    def simple_system(self):
        """Create simple test system."""

        def length_fn(s: String) -> float:
            return min(1.0, len(s.tokens) / 10)

        return VectorSystem([FunctionalStructure(length_fn, name="length")])

    def test_homogenized_vs_diverse_deviance(self, simple_system):
        """Test that homogenized strings have lower expected deviance."""
        # Homogenized: all similar length
        homogenized = [String(tokens=("abc",)) for _ in range(20)]

        # Diverse: varying lengths
        diverse = [String(tokens=tuple("x" for _ in range(i))) for i in range(1, 21)]

        homo_metrics = compute_homogenization_metrics(simple_system, homogenized)
        diverse_metrics = compute_homogenization_metrics(simple_system, diverse)

        # Homogenized should have lower variance
        assert homo_metrics.deviance_variance < diverse_metrics.deviance_variance

    def test_more_strings_stable_metrics(self, simple_system):
        """Test that more strings give stable metrics."""
        base_string = String(tokens=("test",))

        # Small sample
        small = [base_string for _ in range(5)]
        small_metrics = compute_homogenization_metrics(simple_system, small)

        # Large sample
        large = [base_string for _ in range(100)]
        large_metrics = compute_homogenization_metrics(simple_system, large)

        # Both should have similar expected deviance (all identical)
        assert small_metrics.expected_deviance == pytest.approx(
            large_metrics.expected_deviance, abs=0.1
        )
