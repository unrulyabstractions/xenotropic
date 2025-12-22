"""
Trajectory reward functions for xeno-reproduction.

Section 5.3: Trajectory-level formulation.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np

from xenotechnics.common import String, AbstractSystem
from xenotechnics.systems.vector_system import VectorOrientation


def trajectory_reward(
    system: AbstractSystem,
    trajectory: String,
    reference_strings: List[String],
    lambda_d: float = 1.0,
    lambda_f: float = 1.0,
    lambda_c: float = 1.0
) -> float:
    """
    Compute reward for a trajectory in xeno-reproduction.

    Paper (Section 5.3, Equation 24):
    "Trajectory reward combines diversity, fairness, and concentration"

    The reward encourages trajectories that:
    - Have high deviance from reference distribution (diversity)
    - Don't over-concentrate on any single structure (fairness)
    - Maintain high core entropy (concentration)

    Args:
        system: The system to evaluate
        trajectory: String to score
        reference_strings: Reference distribution for core computation
        lambda_d: Weight for diversity component
        lambda_f: Weight for fairness component
        lambda_c: Weight for concentration component

    Returns:
        Scalar reward (higher is better)
    """
    # Compute core from reference distribution
    core_compliance = system.core(reference_strings)

    # Compute trajectory compliance
    trajectory_compliance = system.compliance(trajectory)

    # Diversity: deviance from core
    orientation = VectorOrientation(
        trajectory_compliance,
        core_compliance,
        difference_operator=system.difference_operator
    )
    diversity = orientation.deviance()

    # Fairness: negative max component (promotes uniformity)
    trajectory_array = trajectory_compliance.to_array()
    fairness = -float(np.max(trajectory_array))

    # Concentration: entropy of trajectory compliance
    trajectory_normalized = trajectory_array / (trajectory_array.sum() + 1e-10)
    trajectory_normalized = np.clip(trajectory_normalized, 1e-10, 1.0)
    concentration = float(-np.sum(trajectory_normalized * np.log(trajectory_normalized)))

    # Combine components
    reward = lambda_d * diversity + lambda_f * fairness + lambda_c * concentration

    return reward
