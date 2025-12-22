"""
Pure statistical utility functions.

Section 4: Statistics for homogenization detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .orientation import Orientation
from .string import String
from .system import AbstractSystem

if TYPE_CHECKING:
    from xenotechnics.trees.tree import TreeNode


def expected_deviance(system: AbstractSystem, root: TreeNode, prompt: String) -> float:
    """
    Compute expected deviance E[∂_n] weighted by trajectory probabilities.

    Paper (Section 4, Equation 12):
    "The expected deviance measures average deviation from core"

    Args:
        system: The system to evaluate
        root: Root node of the tree containing trajectory distribution
        prompt: Prompt string (for computing conditional probabilities)

    Returns:
        Expected deviance weighted by trajectory probabilities
    """
    # Get all trajectory nodes and their probabilities using TreeNode methods
    trajectory_nodes = root.get_trajectory_nodes()

    if not trajectory_nodes:
        return 0.0

    probabilities = root.get_conditional_probabilities(
        trajectory_nodes, prompt, normalize=True
    )

    # Compute core
    core_compliance = system.core(root, prompt)

    # Compute weighted deviance
    weighted_deviance = 0.0
    for traj_node, prob in zip(trajectory_nodes, probabilities):
        compliance = system.compliance(traj_node.string)

        orientation = Orientation(
            compliance,
            core_compliance,
            difference_operator=system.difference_operator,
        )
        weighted_deviance += prob * orientation.deviance()

    return float(weighted_deviance)


def deviance_variance(system: AbstractSystem, root: TreeNode, prompt: String) -> float:
    """
    Compute variance of deviance Var[∂_n] weighted by trajectory probabilities.

    Paper (Section 4, Equation 13):
    "The deviance variance measures consistency of deviations"

    Args:
        system: The system to evaluate
        root: Root node of the tree containing trajectory distribution
        prompt: Prompt string (for computing conditional probabilities)

    Returns:
        Variance of deviance weighted by trajectory probabilities
    """
    # Get all trajectory nodes and their probabilities using TreeNode methods
    trajectory_nodes = root.get_trajectory_nodes()

    if not trajectory_nodes:
        return 0.0

    probabilities = root.get_conditional_probabilities(
        trajectory_nodes, prompt, normalize=True
    )

    # Compute core
    core_compliance = system.core(root, prompt)

    # Compute deviances
    deviances = []
    for traj_node in trajectory_nodes:
        compliance = system.compliance(traj_node.string)
        from .orientation import Orientation

        orientation = Orientation(
            compliance1=compliance,
            compliance2=core_compliance,
            difference_operator=system.difference_operator,
        )
        deviances.append(orientation.deviance())

    deviances_array = np.array(deviances)

    # Weighted mean
    mean_deviance = np.sum(probabilities * deviances_array)

    # Weighted variance
    variance = np.sum(probabilities * (deviances_array - mean_deviance) ** 2)

    return float(variance)
