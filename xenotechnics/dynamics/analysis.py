"""
Analysis and evaluation of trajectory dynamics.

Section 3.5: Analyzing dynamics evolution and identifying critical points.
"""

from __future__ import annotations
from typing import Dict, List

import numpy as np

from .base import AbstractDynamics


def analyze_evolution(dynamics: AbstractDynamics) -> Dict[str, any]:
    """
    Analyze the dynamics evolution.

    Args:
        dynamics: Dynamics object to analyze

    Returns:
        Dictionary with analysis metrics including:
        - num_steps: Number of steps in trajectory
        - x_phi_stats: Statistics about continuation cores
        - y_phi_stats: Statistics about path orientation from root
        - z_phi_stats: Statistics about trajectory orientation from position
    """
    x_phis, y_phis, z_phis = dynamics.get_evolution()

    if len(x_phis) == 0:
        return {}

    def compute_norm_stats(arrays: np.ndarray) -> Dict[str, float]:
        """Compute norm statistics for array sequence."""
        norms = [np.linalg.norm(arr) for arr in arrays]
        return {
            'start': float(norms[0]),
            'end': float(norms[-1]),
            'mean': float(np.mean(norms)),
            'std': float(np.std(norms)),
            'max': float(np.max(norms)),
            'min': float(np.min(norms)),
        }

    return {
        'num_steps': len(dynamics.states),
        'x_phi_stats': compute_norm_stats(x_phis),
        'y_phi_stats': compute_norm_stats(y_phis),
        'z_phi_stats': compute_norm_stats(z_phis),
    }


def identify_critical_steps(
    dynamics: AbstractDynamics,
    threshold: float = 0.5,
    component: str = 'y_phi'
) -> List[int]:
    """
    Identify critical steps where dynamics change significantly.

    Paper (Section 7):
    "Certain words may act as 'branching points' where the dynamics
    bifurcate dramatically. Identifying these could reveal where
    diversity is most at stake during generation."

    Args:
        dynamics: The trajectory dynamics
        threshold: Threshold for significant change
        component: Which component to analyze ('x_phi', 'y_phi', or 'z_phi')

    Returns:
        List of step indices where significant changes occur
    """
    x_phis, y_phis, z_phis = dynamics.get_evolution()

    # Select component to analyze
    if component == 'x_phi':
        arrays = x_phis
    elif component == 'y_phi':
        arrays = y_phis
    elif component == 'z_phi':
        arrays = z_phis
    else:
        raise ValueError(f"Unknown component: {component}")

    if len(arrays) < 2:
        return []

    critical_steps = []

    for i in range(1, len(arrays)):
        # Measure change in component
        delta = np.linalg.norm(arrays[i] - arrays[i - 1])

        if delta > threshold:
            critical_steps.append(i)

    return critical_steps


def compute_trajectory_stability(dynamics: AbstractDynamics) -> float:
    """
    Compute stability measure for trajectory dynamics.

    Lower values indicate more stable (less variable) dynamics.

    Args:
        dynamics: The trajectory dynamics

    Returns:
        Stability score (variance of z_phi norms)
    """
    _, _, z_phis = dynamics.get_evolution()

    if len(z_phis) < 2:
        return 0.0

    # Variance of z_phi norms (should decrease to zero)
    z_norms = np.array([np.linalg.norm(z) for z in z_phis])
    stability = float(np.var(z_norms))

    return stability
