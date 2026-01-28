"""
Exploration framework for trajectory collection and analysis.

Provides:
- ModelRunner: TransformerLens-based model interface
- TrajectoryCollector: Sampling trajectories with probabilities
- CoreEstimator: Computing cores and deviances from trajectories
"""

from .analysis import (
    CoreEstimationResult,
    CoreEstimator,
    CoreEstimatorConfig,
    StructureScore,
    Trajectory,
)
from .collection import (
    CollectedTrajectory,
    CollectionProgress,
    CollectionResult,
    CollectionStats,
    TrajectoryCollector,
    TrajectoryCollectorConfig,
)
from .common import ModelRunner

__all__ = [
    # Core
    "ModelRunner",
    # Collection
    "CollectedTrajectory",
    "CollectionProgress",
    "CollectionResult",
    "CollectionStats",
    "TrajectoryCollector",
    "TrajectoryCollectorConfig",
    # Analysis
    "CoreEstimator",
    "CoreEstimatorConfig",
    "CoreEstimationResult",
    "StructureScore",
    "Trajectory",
]
