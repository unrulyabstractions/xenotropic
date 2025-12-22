"""
Vector-based system implementation and compliance.

A system where structures are represented directly as vectors,
enabling efficient mathematical operations.
"""

from __future__ import annotations

from typing import Callable, List, Optional, TYPE_CHECKING

import numpy as np

from xenotechnics.common import (
    AbstractDifferenceOperator,
    AbstractScoreOperator,
    AbstractStructure,
    AbstractSystem,
    AbstractSystemCompliance,
    Orientation,
    String,
)

if TYPE_CHECKING:
    from xenotechnics.operators import (
        VectorScoreOperator,
        VectorDifferenceOperator,
    )


class VectorSystemCompliance(AbstractSystemCompliance):
    """
    Vector-based system compliance implementation.

    Stores compliance as a numpy array.
    """

    def __init__(
        self,
        system: AbstractSystem,
        compliance_vector: np.ndarray,
        string: String = None,
    ):
        """
        Initialize vector compliance.

        Args:
            system: The system that computed this compliance
            compliance_vector: Array of compliance values
            string: Optional string this compliance is for
        """
        super().__init__(system, string)
        self.vector = np.array(compliance_vector)

        if len(self.vector) != len(system):
            raise ValueError(
                f"Compliance vector length {len(self.vector)} must match system size {len(system)}"
            )

    def to_array(self) -> np.ndarray:
        """Get compliance as numpy array."""
        return self.vector.copy()

    def __repr__(self) -> str:
        return f"VectorSystemCompliance(n={len(self)}, mean={self.vector.mean():.3f})"


class VectorSystem(AbstractSystem):
    """
    System implementation composing multiple structures into a vector.

    Evaluates each structure individually and combines their compliances
    into a vector representation.

    Paper (Section 3.2):
    "A system is a collection of structures with compliance vector
    Λ_n(x) = (α_1(x), ..., α_n(x))"
    """

    def __init__(
        self,
        structures: List[AbstractStructure],
        score_operator: Optional[AbstractScoreOperator] = None,
        difference_operator: Optional[AbstractDifferenceOperator] = None,
    ):
        """
        Initialize VectorSystem.

        Args:
            structures: List of structures to compose into a system
            score_operator: Score operator (defaults to L2SquaredScoreOperator)
            difference_operator: Difference operator (defaults to L2SquaredDifferenceOperator)
        """
        # Import operators here to avoid circular import at module level
        from xenotechnics.operators import (
            VectorScoreOperator,
            VectorDifferenceOperator,
            L2SquaredScoreOperator,
            L2SquaredDifferenceOperator,
        )

        if not structures:
            raise ValueError("VectorSystem requires at least one structure")

        # Use default squared L2 operators if not provided
        if score_operator is None:
            score_operator = L2SquaredScoreOperator()
        if difference_operator is None:
            difference_operator = L2SquaredDifferenceOperator()

        # Type check: must be compatible operators (any AbstractScoreOperator/AbstractDifferenceOperator)
        if not isinstance(score_operator, AbstractScoreOperator):
            raise TypeError(
                f"score_operator must be AbstractScoreOperator, got {type(score_operator)}"
            )
        if not isinstance(difference_operator, AbstractDifferenceOperator):
            raise TypeError(
                f"difference_operator must be AbstractDifferenceOperator, got {type(difference_operator)}"
            )

        self.structures = structures
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
        """Compute Λ_n(x) = (α_1(x), ..., α_n(x))."""
        compliance_values = [structure.compliance(string) for structure in self.structures]
        compliance_vector = np.array(compliance_values)

        return VectorSystemCompliance(
            system=self, compliance_vector=compliance_vector, string=string
        )

    def structure_names(self) -> List[str]:
        return [structure.name for structure in self.structures]

    def __len__(self) -> int:
        return len(self.structures)

    def compute_core(self, trajectories, probabilities) -> AbstractSystemCompliance:
        """Compute system core using probability-weighted average."""
        return compute_core_from_trajectories(self, trajectories, probabilities)

    def __repr__(self) -> str:
        return f"VectorSystem({len(self.structures)} structures)"


class VectorOrientation(Orientation):
    """
    Vector-based orientation with explicit vector representation.

    Extends Orientation with to_array() method for accessing
    the orientation vector directly.
    """

    def __init__(
        self,
        compliance1: VectorSystemCompliance,
        compliance2: VectorSystemCompliance,
        difference_operator: AbstractDifferenceOperator,
    ):
        """
        Initialize VectorOrientation.

        Args:
            compliance1: First vector compliance
            compliance2: Second vector compliance
            difference_operator: Difference operator
        """
        super().__init__(compliance1, compliance2, difference_operator)

        # Compute orientation vector
        self.vector = compliance1.to_array() - compliance2.to_array()

    def to_array(self) -> np.ndarray:
        """Get orientation as numpy array."""
        return self.vector.copy()

    def __repr__(self) -> str:
        norm = np.linalg.norm(self.vector)
        return f"VectorOrientation(||θ||={norm:.3f}, dim={len(self)})"


# Utility functions for VectorSystemCompliance


def core_entropy(core_compliance: VectorSystemCompliance) -> float:
    """
    Compute entropy of system core H(⟨Λ_n⟩).

    Paper (Section 4, Equation 14):
    "The core entropy measures concentration of the distribution"

    Args:
        core_compliance: Core system compliance (should be VectorSystemCompliance)

    Returns:
        Entropy of core
    """
    core_vector = core_compliance.to_array()

    # Normalize to probability distribution
    p = core_vector / (core_vector.sum() + 1e-10)
    p = np.clip(p, 1e-10, 1.0)  # Avoid log(0)

    return float(-np.sum(p * np.log(p)))


def compute_core_from_trajectories(
    system: AbstractSystem, trajectories: List, probabilities: np.ndarray
) -> VectorSystemCompliance:
    """
    Compute system core ⟨Λ_n⟩ = E_p[Λ_n(x)] from trajectories and probabilities.

    Paper (Section 3.3, Equation 5):
    "The system core is the expected compliance:
    ⟨Λ_n⟩ = E_x~p[Λ_n(x)]"

    This is the probability-weighted average of compliances.

    Args:
        system: The system to compute compliance with
        trajectories: List of Trajectory objects
        probabilities: Array of probabilities (must sum to 1)

    Returns:
        Core as VectorSystemCompliance with string=None
    """
    if not trajectories:
        raise ValueError("Cannot compute core from empty trajectory list")

    if len(trajectories) != len(probabilities):
        raise ValueError("Number of trajectories must match number of probabilities")

    # Compute compliance for each trajectory
    vectors = []
    for traj in trajectories:
        compliance = system.compliance(traj)
        vectors.append(compliance.to_array())

    vectors_array = np.array(vectors)  # Shape: (n_trajectories, n_structures)

    # Weighted average using probabilities
    core_vector = np.average(vectors_array, axis=0, weights=probabilities)

    return VectorSystemCompliance(
        system=system,
        compliance_vector=core_vector,
        string=None,  # Core has no associated string
    )
