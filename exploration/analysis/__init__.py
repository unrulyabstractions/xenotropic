"""
Analysis module for computing cores and statistics from trajectories.
"""

from .core_estimator import (
    CoreEstimationResult,
    CoreEstimator,
    CoreEstimatorConfig,
    StructureScore,
    Trajectory,
)

__all__ = [
    "CoreEstimationResult",
    "CoreEstimator",
    "CoreEstimatorConfig",
    "StructureScore",
    "Trajectory",
]
