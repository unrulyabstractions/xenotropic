"""
Singleton system implementation.

A simple linear vector system that wraps a single FunctionalStructure.
Useful for quick prototyping and single-structure systems.
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np

from xenotechnics.common import (
    AbstractDifferenceOperator,
    AbstractScoreOperator,
    AbstractSystem,
    AbstractSystemCompliance,
    FunctionalStructure,
    String,
)

from .vector_system import VectorSystemCompliance, compute_core_from_trajectories


class SingletonSystem(AbstractSystem):
    """
    System with a single FunctionalStructure.

    This is a convenience class for creating systems from a single
    compliance function without explicitly creating a structure class.

    Example:
        >>> def my_compliance(string: String) -> float:
        ...     return len(string) / 100
        >>> system = SingletonSystem(
        ...     compliance_fn=my_compliance,
        ...     name="length_normalized",
        ...     score_operator=L2ScoreOperator(),
        ...     difference_operator=L2DifferenceOperator()
        ... )
    """

    def __init__(
        self,
        compliance_fn: Callable[[String], float],
        score_operator: AbstractScoreOperator,
        difference_operator: AbstractDifferenceOperator,
        name: str = "singleton",
        description: str = "Single structure system",
    ):
        """
        Initialize SingletonSystem.

        Args:
            compliance_fn: Function that takes a String and returns compliance in [0, 1]
            score_operator: Score operator for this system
            difference_operator: Difference operator for this system
            name: System name
            description: System description
        """
        self.structure = FunctionalStructure(
            compliance_fn=compliance_fn, name=name, description=description
        )
        self._score_operator = score_operator
        self._difference_operator = difference_operator

    @property
    def score_operator(self) -> AbstractScoreOperator:
        """Score operator for this system."""
        return self._score_operator

    @property
    def difference_operator(self) -> AbstractDifferenceOperator:
        """Difference operator for this system."""
        return self._difference_operator

    def compliance(self, string: String) -> AbstractSystemCompliance:
        """
        Compute compliance using the single structure.

        Returns:
            VectorSystemCompliance with single element
        """
        compliance_value = self.structure.compliance(string)
        return VectorSystemCompliance(
            system=self, compliance_vector=np.array([compliance_value]), string=string
        )

    def structure_names(self) -> List[str]:
        """Get structure name."""
        return [self.structure.name]

    def __len__(self) -> int:
        """Number of structures (always 1)."""
        return 1

    def compute_core(self, trajectories, probabilities) -> AbstractSystemCompliance:
        """Compute system core using probability-weighted average."""
        return compute_core_from_trajectories(self, trajectories, probabilities)

    def __repr__(self) -> str:
        return f"SingletonSystem(name='{self.structure.name}')"
