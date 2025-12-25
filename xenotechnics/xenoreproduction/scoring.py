"""
Scoring functions for xeno-reproduction interventions.

Section 5.1: Distribution-level scoring.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from xenotechnics.common import AbstractSystem, Orientation, String
from xenotechnics.systems.vector_system import core_entropy

from .data import InterventionScores


def score_diversity(system: AbstractSystem, strings: Iterable[String]) -> float:
    """
    Score diversity ρ_d based on deviance.

    Paper (Section 5.1.1, Equation 18):
    "Diversity score ρ_d = E[∂_n] + Var[∂_n]"

    Args:
        system: The system to evaluate
        strings: Collection of strings

    Returns:
        Diversity score (higher is better)
    """
    strings_list = list(strings)

    if not strings_list:
        return 0.0

    # Compute core and orientations using uniform probabilities
    n = len(strings_list)
    uniform_probs = np.ones(n) / n
    core_compliance = system.compute_core(strings_list, uniform_probs)
    deviances = []
    for s in strings_list:
        compliance = system.compliance(s)
        orientation = Orientation(
            compliance, core_compliance, difference_operator=system.difference_operator
        )
        deviances.append(orientation.deviance())

    # Compute diversity metrics
    deviances_array = np.array(deviances)
    exp_dev = float(np.mean(deviances_array))
    dev_var = float(np.var(deviances_array))

    return exp_dev + dev_var


def score_fairness(system: AbstractSystem, strings: Iterable[String]) -> float:
    """
    Score fairness ρ_f based on uniformity of core.

    Paper (Section 5.1.2, Equation 19):
    "Fairness score ρ_f = -max_i(⟨Λ_n⟩_i)"

    Args:
        system: The system to evaluate
        strings: Collection of strings

    Returns:
        Fairness score (higher is better, less negative)
    """
    strings_list = list(strings)

    if not strings_list:
        return 0.0

    n = len(strings_list)
    uniform_probs = np.ones(n) / n
    core_compliance = system.compute_core(strings_list, uniform_probs)
    core_array = core_compliance.to_array()

    # Negative of maximum component (promotes uniformity)
    return -float(np.max(core_array))


def score_concentration(system: AbstractSystem, strings: Iterable[String]) -> float:
    """
    Score concentration ρ_c based on core entropy.

    Paper (Section 5.1.3, Equation 20):
    "Concentration score ρ_c = H(⟨Λ_n⟩)"

    Args:
        system: The system to evaluate
        strings: Collection of strings

    Returns:
        Concentration score (higher entropy is better)
    """
    strings_list = list(strings)

    if not strings_list:
        return 0.0

    n = len(strings_list)
    uniform_probs = np.ones(n) / n
    core_compliance = system.compute_core(strings_list, uniform_probs)

    # Core entropy (only for vector systems)
    try:
        return core_entropy(core_compliance)
    except (AttributeError, TypeError):
        return 0.0


def score_intervention(
    system: AbstractSystem,
    strings: Iterable[String],
    lambda_d: float = 1.0,
    lambda_f: float = 1.0,
    lambda_c: float = 1.0,
) -> InterventionScores:
    """
    Compute all intervention scores.

    Paper (Section 5.1, Equation 21):
    "Total score: ρ = λ_d·ρ_d + λ_f·ρ_f + λ_c·ρ_c"

    Args:
        system: The system to evaluate
        strings: Collection of strings
        lambda_d: Weight for diversity score
        lambda_f: Weight for fairness score
        lambda_c: Weight for concentration score

    Returns:
        InterventionScores with all component and total scores
    """
    diversity = score_diversity(system, strings)
    fairness = score_fairness(system, strings)
    concentration = score_concentration(system, strings)

    total = lambda_d * diversity + lambda_f * fairness + lambda_c * concentration

    return InterventionScores(
        diversity=diversity, fairness=fairness, concentration=concentration, total=total
    )
