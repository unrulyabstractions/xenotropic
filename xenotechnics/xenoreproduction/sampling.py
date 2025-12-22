"""
Xeno-distribution sampling.

Section 5.2: Sampling from diversity-promoting distributions.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np
from scipy.special import softmax

from xenotechnics.common import String, AbstractSystem, Orientation


def xeno_distribution(
    system: AbstractSystem,
    strings: List[String],
    base_probs: np.ndarray,
    temperature: float = 1.0
) -> np.ndarray:
    """
    Compute xeno-reproduction distribution p^(xeno).

    Paper (Section 5.2, Equation 22):
    "The xeno-distribution reweights the base distribution to promote diversity"

    Args:
        system: The system to evaluate
        strings: List of strings to reweight
        base_probs: Base probability distribution
        temperature: Temperature for softmax (higher = more uniform)

    Returns:
        Xeno-distribution probabilities (normalized)
    """
    if len(strings) != len(base_probs):
        raise ValueError("Number of strings must match number of probabilities")

    if not strings:
        return np.array([])

    # Compute per-string deviances as proxy for diversity
    core_compliance = system.core(strings)
    deviances = []
    for s in strings:
        compliance = system.compliance(s)
        orientation = Orientation(
            compliance,
            core_compliance,
            difference_operator=system.difference_operator
        )
        deviances.append(orientation.deviance())
    deviances = np.array(deviances)

    # Xeno-distribution: upweight strings with high deviance
    logits = np.log(base_probs + 1e-10) + (deviances / temperature)
    xeno_probs = softmax(logits)

    return xeno_probs


def sample_xeno_trajectory(
    system: AbstractSystem,
    trajectory: String,
    reference_strings: Optional[List[String]] = None
) -> float:
    """
    Compute xeno-probability for a trajectory.

    Paper (Section 5.3, Equation 23):
    "Trajectory xeno-probability is based on its deviance from reference"

    Args:
        system: The system to evaluate
        trajectory: String to score
        reference_strings: Reference strings for core (default: just trajectory)

    Returns:
        Xeno-probability score (higher deviance = higher probability)
    """
    if reference_strings is None:
        reference_strings = [trajectory]

    # Compute deviance as measure of diversity contribution
    core_compliance = system.core(reference_strings)
    trajectory_compliance = system.compliance(trajectory)
    orientation = Orientation(
        trajectory_compliance,
        core_compliance,
        difference_operator=system.difference_operator
    )
    deviance = orientation.deviance()

    # Higher deviance = higher xeno-probability
    return float(deviance)
