"""
Entropic systems for measuring deviation from reference distributions.

These systems measure excess (over-representation) and deficit (under-representation)
relative to a baseline distribution using information-theoretic measures.
"""

from __future__ import annotations

from typing import List
import numpy as np

from xenotechnics.common import (
    AbstractSystem,
    AbstractSystemCompliance,
    AbstractScoreOperator,
    AbstractDifferenceOperator,
    String
)
from .vector_system import VectorSystemCompliance, compute_core_from_trajectories


class ExcessSystem(AbstractSystem):
    """
    System that measures excess compliance relative to a baseline.

    Paper (Appendix C.2):
    "Excess measures how much a string over-represents structures
    relative to a reference distribution."

    For each structure, computes:
    excess_i(x) = max(0, α_i(x) - baseline_i)

    This is useful for:
    - Detecting mode collapse (too much concentration)
    - Measuring diversity violations
    - Identifying over-represented structures
    """

    def __init__(
        self,
        base_system: AbstractSystem,
        baseline: np.ndarray = None
    ):
        """
        Initialize ExcessSystem.

        Args:
            base_system: The underlying system to measure
            baseline: Reference compliance values (default: uniform distribution)
        """
        self.base_system = base_system
        self.n_structures = len(base_system)
        self.score_operator = base_system.score_operator
        self.difference_operator = base_system.difference_operator

        if baseline is None:
            # Default: uniform baseline (no structure should dominate)
            self.baseline = np.ones(self.n_structures) / self.n_structures
        else:
            self.baseline = np.array(baseline)
            if len(self.baseline) != self.n_structures:
                raise ValueError(
                    f"Baseline length {len(self.baseline)} must match "
                    f"number of structures {self.n_structures}"
                )

    def compliance(self, string: String) -> AbstractSystemCompliance:
        """
        Compute excess compliance: max(0, α_i(x) - baseline_i).

        Returns:
            VectorSystemCompliance with excess values (zero where no excess)
        """
        # Get base compliance
        base_compliance = self.base_system.compliance(string)
        base_vector = base_compliance.to_array()

        # Compute excess: positive part of (compliance - baseline)
        excess = np.maximum(0.0, base_vector - self.baseline)

        return VectorSystemCompliance(
            system=self,
            compliance_vector=excess,
            string=string
        )

    def structure_names(self) -> List[str]:
        """Structure names prefixed with 'excess_'."""
        return [f"excess_{name}" for name in self.base_system.structure_names()]

    def __len__(self) -> int:
        """Number of structures."""
        return self.n_structures

    def compute_core(self, trajectories, probabilities) -> AbstractSystemCompliance:
        """Compute system core using probability-weighted average."""
        return compute_core_from_trajectories(self, trajectories, probabilities)

    def set_baseline(self, baseline: np.ndarray):
        """
        Update the baseline reference.

        Args:
            baseline: New baseline values
        """
        if len(baseline) != self.n_structures:
            raise ValueError("Baseline length must match number of structures")
        self.baseline = np.array(baseline)

    def set_baseline_from_core(
        self,
        tree_root,
        prompt: String
    ):
        """
        Update baseline to be the core of the base system.

        This sets the baseline to the expected compliance over the tree,
        so excess measures deviation above this expected value.

        Args:
            tree_root: Root of generation tree
            prompt: Prompt string
        """
        core_compliance = self.base_system.core(tree_root, prompt)
        self.baseline = core_compliance.to_array()

    def __repr__(self) -> str:
        return (
            f"ExcessSystem({self.n_structures} structures, "
            f"baseline_mean={self.baseline.mean():.3f})"
        )


class DeficitSystem(AbstractSystem):
    """
    System that measures deficit compliance relative to a baseline.

    Paper (Appendix C.2):
    "Deficit measures how much a string under-represents structures
    relative to a reference distribution."

    For each structure, computes:
    deficit_i(x) = max(0, baseline_i - α_i(x))

    This is useful for:
    - Identifying underrepresented structures
    - Guiding interventions to promote fairness
    - Detecting which structures need more representation
    """

    def __init__(
        self,
        base_system: AbstractSystem,
        baseline: np.ndarray = None
    ):
        """
        Initialize DeficitSystem.

        Args:
            base_system: The underlying system to measure
            baseline: Reference compliance values (default: uniform distribution)
        """
        self.base_system = base_system
        self.n_structures = len(base_system)
        self.score_operator = base_system.score_operator
        self.difference_operator = base_system.difference_operator

        if baseline is None:
            # Default: uniform baseline (all structures should be represented equally)
            self.baseline = np.ones(self.n_structures) / self.n_structures
        else:
            self.baseline = np.array(baseline)
            if len(self.baseline) != self.n_structures:
                raise ValueError(
                    f"Baseline length {len(self.baseline)} must match "
                    f"number of structures {self.n_structures}"
                )

    def compliance(self, string: String) -> AbstractSystemCompliance:
        """
        Compute deficit compliance: max(0, baseline_i - α_i(x)).

        Returns:
            VectorSystemCompliance with deficit values (zero where no deficit)
        """
        # Get base compliance
        base_compliance = self.base_system.compliance(string)
        base_vector = base_compliance.to_array()

        # Compute deficit: positive part of (baseline - compliance)
        deficit = np.maximum(0.0, self.baseline - base_vector)

        return VectorSystemCompliance(
            system=self,
            compliance_vector=deficit,
            string=string
        )

    def structure_names(self) -> List[str]:
        """Structure names prefixed with 'deficit_'."""
        return [f"deficit_{name}" for name in self.base_system.structure_names()]

    def __len__(self) -> int:
        """Number of structures."""
        return self.n_structures

    def compute_core(self, trajectories, probabilities) -> AbstractSystemCompliance:
        """Compute system core using probability-weighted average."""
        return compute_core_from_trajectories(self, trajectories, probabilities)

    def set_baseline(self, baseline: np.ndarray):
        """
        Update the baseline reference.

        Args:
            baseline: New baseline values
        """
        if len(baseline) != self.n_structures:
            raise ValueError("Baseline length must match number of structures")
        self.baseline = np.array(baseline)

    def set_baseline_from_core(
        self,
        tree_root,
        prompt: String
    ):
        """
        Update baseline to be the core of the base system.

        This sets the baseline to the expected compliance over the tree,
        so deficit measures deviation below this expected value.

        Args:
            tree_root: Root of generation tree
            prompt: Prompt string
        """
        core_compliance = self.base_system.core(tree_root, prompt)
        self.baseline = core_compliance.to_array()

    def __repr__(self) -> str:
        return (
            f"DeficitSystem({self.n_structures} structures, "
            f"baseline_mean={self.baseline.mean():.3f})"
        )
