"""
Trajectory collection module.

Provides TrajectoryCollector for sampling trajectories from language models.
"""

from .trajectory_collector import (
    CollectedTrajectory,
    CollectionProgress,
    CollectionResult,
    CollectionStats,
    TrajectoryCollector,
    TrajectoryCollectorConfig,
)

__all__ = [
    "CollectedTrajectory",
    "CollectionProgress",
    "CollectionResult",
    "CollectionStats",
    "TrajectoryCollector",
    "TrajectoryCollectorConfig",
]
