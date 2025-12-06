"""
Core estimation from trajectories.

Computes probability-weighted cores and deviances for structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


class Trajectory(Protocol):
    """Any object with text and probability attributes."""

    text: str
    probability: float
    log_probability: float


@dataclass
class CoreEstimatorConfig:
    """Configuration for core estimation."""

    use_log_space: bool = True  # Use log-space normalization to avoid underflow


@dataclass
class StructureScore:
    """Scores for a single structure across all trajectories."""

    structure: str
    scores: list[float]
    core: float
    expected_deviance: float
    var_deviance: float


@dataclass
class CoreEstimationResult:
    """Result of core estimation."""

    structures: list[StructureScore]
    probabilities: list[float]

    @property
    def aggregate_core(self) -> float:
        """Mean core across structures."""
        return float(np.mean([s.core for s in self.structures]))

    @property
    def aggregate_deviance(self) -> float:
        """Mean deviance across structures."""
        return float(np.mean([s.expected_deviance for s in self.structures]))


class CoreEstimator:
    """
    Estimate cores and deviances from trajectories.

    Core = E[score] = sum(prob_i * score_i)
    Expected deviance = E[|score - core|]
    """

    def __init__(self, config: CoreEstimatorConfig | None = None):
        self.config = config or CoreEstimatorConfig()

    def estimate(
        self,
        trajectories: list[Trajectory],
        structures: list[str],
        scorer_factory: Callable[[str], Callable[[str], float]],
        context_prefix: str = "",
    ) -> CoreEstimationResult:
        """
        Estimate cores for structures from trajectories.

        Args:
            trajectories: List of trajectories with probabilities
            structures: List of structure questions
            scorer_factory: Creates a scorer for a structure: scorer_factory(structure) -> scorer
            context_prefix: Prefix to prepend to trajectory text (e.g., the prompt)

        Returns:
            CoreEstimationResult with scores, cores, and deviances
        """
        probs = self._get_probabilities(trajectories)

        structure_scores = []
        for structure in structures:
            scorer = scorer_factory(structure)
            scores = [scorer(context_prefix + t.text) for t in trajectories]
            scores_array = np.array(scores)

            # Core = probability-weighted mean
            core = float(np.sum(scores_array * probs))

            # Deviance = |score - core|
            deviances = np.abs(scores_array - core)
            expected_deviance = float(np.sum(deviances * probs))
            var_deviance = float(np.sum((deviances - expected_deviance) ** 2 * probs))

            structure_scores.append(
                StructureScore(
                    structure=structure,
                    scores=scores,
                    core=core,
                    expected_deviance=expected_deviance,
                    var_deviance=var_deviance,
                )
            )

        return CoreEstimationResult(
            structures=structure_scores,
            probabilities=probs.tolist(),
        )

    def _get_probabilities(self, trajectories: list[Trajectory]) -> np.ndarray:
        """Get normalized probabilities from trajectories."""
        if self.config.use_log_space:
            log_probs = np.array([t.log_probability for t in trajectories])
            log_probs = log_probs - np.max(log_probs)  # Numerical stability
            probs = np.exp(log_probs)
        else:
            probs = np.array([t.probability for t in trajectories])

        return probs / probs.sum()
