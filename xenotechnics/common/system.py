"""
Abstract system interface.

Section 3.2: Systems as collections of structures
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from .compliance import AbstractSystemCompliance
from .operator import AbstractDifferenceOperator, AbstractScoreOperator
from .string import String

if TYPE_CHECKING:
    from xenotechnics.trees.tree import TreeNode


class AbstractSystem(ABC):
    """
    Abstract base class for systems.

    Paper (Section 3.2, Equation 3):
    "We call a system the collection of structures of interest. We define
    the system compliance as a vector of compliances across particular
    structures: Λ_n(x) := (α_1(x), ..., α_n(x))"

    A system is only fully implemented when its operators are defined.
    - ScoreOperator: determines how deviance is computed
    - DifferenceOperator: determines how orientation is computed

    Systems do NOT have orientation() or deviance() methods.
    Those are the Orientation class's responsibility.
    """

    @abstractmethod
    def compliance(self, string: String) -> AbstractSystemCompliance:
        """
        Compute system compliance Λ_n(x) = (α_1(x), ..., α_n(x)).

        Args:
            string: The string to evaluate

        Returns:
            SystemCompliance object
        """
        pass

    @abstractmethod
    def structure_names(self) -> List[str]:
        """Get names of all structures in this system."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Number of structures in this system."""
        pass

    @property
    @abstractmethod
    def score_operator(self) -> AbstractScoreOperator:
        """Score operator for this system."""
        pass

    @property
    @abstractmethod
    def difference_operator(self) -> AbstractDifferenceOperator:
        """Difference operator for this system."""
        pass

    # Core computation (Section 3.3)

    @abstractmethod
    def compute_core(
        self, trajectories: List[String], probabilities: np.ndarray
    ) -> AbstractSystemCompliance:
        """
        Compute system core ⟨Λ_n⟩ = E_p[Λ_n(x)].

        This is probability-weighted average of compliances.

        Args:
            trajectories: List of trajectory strings
            probabilities: Array of probabilities for each trajectory

        Returns:
            Core as SystemCompliance object with string=None
        """
        pass

    def core(
        self,
        tree_root: TreeNode,
        prompt: String,
        trajectories: Optional[List[String]] = None,
    ) -> AbstractSystemCompliance:
        """
        Compute system core from tree and prompt.

        Args:
            tree_root: Root of the tree containing trajectories
            prompt: Prompt string (prefix before generation starts)
            trajectories: Optional list of specific trajectories to compute core over.
                        If None, uses all trajectories in tree.

        Returns:
            Core as SystemCompliance object

        Raises:
            ValueError: If no trajectories found or if specified trajectory not in tree
        """
        if trajectories is None:
            # Get all trajectory nodes from tree
            trajectory_nodes = tree_root.get_trajectory_nodes()

            if not trajectory_nodes:
                raise ValueError("No trajectories found in tree")

            # Extract String objects
            trajectories = [node.string for node in trajectory_nodes]
        else:
            # Find trajectory nodes for the specified trajectories
            trajectory_nodes = []
            for traj in trajectories:
                node = tree_root.find_trajectory_node(traj)
                if node is None:
                    raise ValueError(f"Trajectory not found in tree: {traj}")
                trajectory_nodes.append(node)

        # Get conditional probabilities from tree
        probabilities = tree_root.get_conditional_probabilities(
            trajectory_nodes, prompt, normalize=False
        )

        return self.compute_core(trajectories, probabilities)
