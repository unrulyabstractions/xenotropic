"""
Tests for xenoreproduction data structures.

Tests for xenotechnics/xenoreproduction/data.py
"""

from __future__ import annotations

from xenotechnics.xenoreproduction.data import HomogenizationMetrics, InterventionScores


class TestHomogenizationMetrics:
    """Test HomogenizationMetrics dataclass."""

    def test_creation(self):
        """Test creating HomogenizationMetrics."""
        metrics = HomogenizationMetrics(
            expected_deviance=0.5,
            deviance_variance=0.1,
            core_entropy=1.2,
        )
        assert metrics.expected_deviance == 0.5
        assert metrics.deviance_variance == 0.1
        assert metrics.core_entropy == 1.2

    def test_zero_values(self):
        """Test with zero values (homogenized state)."""
        metrics = HomogenizationMetrics(
            expected_deviance=0.0,
            deviance_variance=0.0,
            core_entropy=0.0,
        )
        assert metrics.expected_deviance == 0.0
        assert metrics.deviance_variance == 0.0
        assert metrics.core_entropy == 0.0

    def test_high_diversity_values(self):
        """Test with high values (diverse state)."""
        metrics = HomogenizationMetrics(
            expected_deviance=0.9,
            deviance_variance=0.5,
            core_entropy=2.0,
        )
        assert metrics.expected_deviance == 0.9
        assert metrics.deviance_variance == 0.5
        assert metrics.core_entropy == 2.0


class TestInterventionScores:
    """Test InterventionScores dataclass."""

    def test_creation(self):
        """Test creating InterventionScores."""
        scores = InterventionScores(
            diversity=0.5,
            fairness=-0.3,
            concentration=1.2,
            total=1.4,
        )
        assert scores.diversity == 0.5
        assert scores.fairness == -0.3
        assert scores.concentration == 1.2
        assert scores.total == 1.4

    def test_negative_fairness(self):
        """Test with negative fairness (expected)."""
        scores = InterventionScores(
            diversity=0.5,
            fairness=-0.8,  # Fairness is negative max component
            concentration=1.0,
            total=0.7,
        )
        assert scores.fairness < 0

    def test_all_zero(self):
        """Test with all zero values."""
        scores = InterventionScores(
            diversity=0.0,
            fairness=0.0,
            concentration=0.0,
            total=0.0,
        )
        assert scores.total == 0.0
