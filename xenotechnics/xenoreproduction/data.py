"""
Data structures for xeno-reproduction results.

Section 4, 5: Metrics and intervention results.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class HomogenizationMetrics:
    """
    Metrics for detecting homogenization.

    Paper (Section 4):
    "Homogenization is detected through: E[∂_n] → 0, Var[∂_n] → 0, H(⟨Λ_n⟩) → 0"

    Attributes:
        expected_deviance: E[∂_n] - average deviation from core
        deviance_variance: Var[∂_n] - consistency of deviations
        core_entropy: H(⟨Λ_n⟩) - concentration of core distribution
    """
    expected_deviance: float
    deviance_variance: float
    core_entropy: float


@dataclass
class InterventionScores:
    """
    Scores for xeno-reproduction interventions.

    Paper (Section 5.1):
    "Interventions are scored on diversity, fairness, and concentration"

    Attributes:
        diversity: ρ_d = E[∂_n] + Var[∂_n] - total diversity score
        fairness: ρ_f = -max_i(⟨Λ_n⟩_i) - uniformity of core
        concentration: ρ_c = H(⟨Λ_n⟩) - entropy of core
        total: Weighted combination of all scores
    """
    diversity: float
    fairness: float
    concentration: float
    total: float
