"""
Trajectory dynamics tracking.

Section 3.5: Tracking how cores and orientations evolve during generation.
"""

from .base import AbstractDynamics, DynamicsState
from .linear import LinearDynamics
from .analysis import analyze_evolution, identify_critical_steps

__all__ = [
    'AbstractDynamics',
    'DynamicsState',
    'LinearDynamics',
    'analyze_evolution',
    'identify_critical_steps',
]
