"""
Orientation representations.

Section 3.4: Orientation θ_n(x) = Λ_n(x) - ⟨Λ_n⟩
"""

from __future__ import annotations

from .compliance import AbstractSystemCompliance
from .operator import AbstractDifferenceOperator


class Orientation:
    """
    Orientation computed using DifferenceOperator.

    Paper (Section 3.4, Equation 8):
    "The orientation of a string is its deviation from the core:
    θ_n(x) = Λ_n(x) - ⟨Λ_n⟩"

    Takes two SystemCompliance objects and computes their difference.
    """

    def __init__(
        self,
        compliance_left: AbstractSystemCompliance,
        compliance_right: AbstractSystemCompliance,
        difference_operator: AbstractDifferenceOperator,
    ):
        """
        Initialize Orientation.

        Args:
            compliance_left: First system compliance (e.g., Λ_n(x))
            compliance_right: Second system compliance (e.g., ⟨Λ_n⟩)
            difference_operator: Operator for computing difference
        """
        if len(compliance_left) != len(compliance_right):
            raise ValueError("Compliances must have same dimension")

        self.compliance_left = compliance_left
        self.compliance_right = compliance_right
        self.difference_operator = difference_operator

    def deviance(self) -> float:
        """
        Compute deviance ∂_n(x) = ||θ_n(x)|| using DifferenceOperator.

        Paper (Section 3.4, Equation 9):
        "The deviance is the magnitude of orientation"

        Returns:
            Scalar deviance score
        """
        return self.difference_operator(self.compliance_left, self.compliance_right)

    def __len__(self) -> int:
        """Dimension of orientation."""
        return len(self.compliance_left)

    def __repr__(self) -> str:
        return f"Orientation(dim={len(self)})"
