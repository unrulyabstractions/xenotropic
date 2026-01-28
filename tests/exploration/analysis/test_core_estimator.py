"""Tests for CoreEstimator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from exploration.analysis import (
    CoreEstimationResult,
    CoreEstimator,
    CoreEstimatorConfig,
    StructureScore,
)


@dataclass
class MockTrajectory:
    """Mock trajectory for testing."""

    text: str
    probability: float
    log_probability: float


class TestCoreEstimatorConfig:
    """Tests for CoreEstimatorConfig."""

    def test_default_values(self):
        config = CoreEstimatorConfig()
        assert config.use_log_space is True

    def test_custom_values(self):
        config = CoreEstimatorConfig(use_log_space=False)
        assert config.use_log_space is False


class TestStructureScore:
    """Tests for StructureScore."""

    def test_creation(self):
        score = StructureScore(
            structure="test",
            scores=[0.5, 0.6, 0.7],
            core=0.6,
            expected_deviance=0.1,
            var_deviance=0.01,
        )
        assert score.structure == "test"
        assert score.core == 0.6


class TestCoreEstimationResult:
    """Tests for CoreEstimationResult."""

    def test_aggregate_core(self):
        result = CoreEstimationResult(
            structures=[
                StructureScore("a", [], 0.4, 0.1, 0.01),
                StructureScore("b", [], 0.6, 0.2, 0.02),
            ],
            probabilities=[0.5, 0.5],
        )
        assert result.aggregate_core == 0.5

    def test_aggregate_deviance(self):
        result = CoreEstimationResult(
            structures=[
                StructureScore("a", [], 0.4, 0.1, 0.01),
                StructureScore("b", [], 0.6, 0.3, 0.02),
            ],
            probabilities=[0.5, 0.5],
        )
        assert result.aggregate_deviance == 0.2


class TestCoreEstimator:
    """Tests for CoreEstimator."""

    def test_init_default_config(self):
        estimator = CoreEstimator()
        assert estimator.config.use_log_space is True

    def test_init_custom_config(self):
        config = CoreEstimatorConfig(use_log_space=False)
        estimator = CoreEstimator(config)
        assert estimator.config.use_log_space is False

    def test_estimate_uniform_scores(self):
        """When all scores are the same, core equals that score and deviance is 0."""
        trajectories = [
            MockTrajectory("a", 0.5, np.log(0.5)),
            MockTrajectory("b", 0.5, np.log(0.5)),
        ]

        def scorer_factory(structure):
            return lambda text: 0.8

        estimator = CoreEstimator(CoreEstimatorConfig(use_log_space=False))
        result = estimator.estimate(trajectories, ["test"], scorer_factory)

        assert len(result.structures) == 1
        assert result.structures[0].core == pytest.approx(0.8)
        assert result.structures[0].expected_deviance == pytest.approx(0.0)

    def test_estimate_varying_scores(self):
        """Test core estimation with varying scores."""
        trajectories = [
            MockTrajectory("a", 0.6, np.log(0.6)),
            MockTrajectory("b", 0.4, np.log(0.4)),
        ]

        scores = {"a": 1.0, "b": 0.0}

        def scorer_factory(structure):
            return lambda text: scores.get(text, 0.5)

        estimator = CoreEstimator(CoreEstimatorConfig(use_log_space=False))
        result = estimator.estimate(trajectories, ["test"], scorer_factory)

        # Core = 0.6 * 1.0 + 0.4 * 0.0 = 0.6
        assert result.structures[0].core == pytest.approx(0.6)

    def test_estimate_with_context_prefix(self):
        """Test that context_prefix is prepended to trajectory text."""
        trajectories = [MockTrajectory("world", 1.0, 0.0)]
        received_texts = []

        def scorer_factory(structure):
            def scorer(text):
                received_texts.append(text)
                return 0.5

            return scorer

        estimator = CoreEstimator(CoreEstimatorConfig(use_log_space=False))
        estimator.estimate(
            trajectories, ["test"], scorer_factory, context_prefix="hello "
        )

        assert received_texts == ["hello world"]

    def test_estimate_log_space(self):
        """Test log-space probability normalization."""
        # Very small probabilities that would underflow
        trajectories = [
            MockTrajectory("a", 1e-100, -230.26),
            MockTrajectory("b", 1e-100, -230.26),
        ]

        def scorer_factory(structure):
            return lambda text: 0.5

        # Log-space should handle this without underflow
        estimator = CoreEstimator(CoreEstimatorConfig(use_log_space=True))
        result = estimator.estimate(trajectories, ["test"], scorer_factory)

        # Should get valid probabilities
        assert sum(result.probabilities) == pytest.approx(1.0)
        assert result.structures[0].core == pytest.approx(0.5)

    def test_estimate_multiple_structures(self):
        """Test estimation with multiple structures."""
        trajectories = [MockTrajectory("x", 1.0, 0.0)]

        structure_scores = {"s1": 0.3, "s2": 0.7}

        def scorer_factory(structure):
            return lambda text: structure_scores[structure]

        estimator = CoreEstimator(CoreEstimatorConfig(use_log_space=False))
        result = estimator.estimate(trajectories, ["s1", "s2"], scorer_factory)

        assert len(result.structures) == 2
        assert result.structures[0].core == pytest.approx(0.3)
        assert result.structures[1].core == pytest.approx(0.7)
        assert result.aggregate_core == pytest.approx(0.5)
