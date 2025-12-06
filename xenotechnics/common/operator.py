"""
Abstract operators for system scores and differences.

Section 3.2, Equation 4
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .compliance import AbstractSystemCompliance


class AbstractScoreOperator(ABC):
    """
    Abstract operator for aggregating system compliance into scalar score.

    Paper (Section 3.2, Equation 4):
    "To enable easy comparisons, we define operators that aggregate
    compliance into scalar system scores: ||Λ_n(x)||_Λ ∈ [0, 1]"
    """

    @abstractmethod
    def __call__(self, compliance: AbstractSystemCompliance) -> float:
        """
        Compute scalar score ||Λ_n(x)||_Λ from compliance.

        Args:
            compliance: System compliance Λ_n(x)

        Returns:
            Scalar score in [0, 1]
        """
        pass


class AbstractDifferenceOperator(ABC):
    """
    Abstract operator for computing difference between two compliances.

    Paper (Section 3.2, Equation 4):
    "...and difference scores: ||Λ_n(x_r) - Λ_n(x_q)||_θ ∈ [0, 1]"

    Takes two system compliances and computes their difference.
    """

    @abstractmethod
    def __call__(
        self, compliance1: AbstractSystemCompliance, compliance2: AbstractSystemCompliance
    ) -> float:
        """
        Compute difference between two system compliances.

        Args:
            compliance1: First compliance (reference)
            compliance2: Second compliance (query)

        Returns:
            Scalar difference score
        """
        pass
