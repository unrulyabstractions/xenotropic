"""
Homogenization metrics computation.

Section 4: Detecting homogenization through statistical measures.
"""

from __future__ import annotations
from typing import Iterable
import numpy as np

from xenotechnics.common import String, AbstractSystem, Orientation
from xenotechnics.systems.vector_system import core_entropy
from .data import HomogenizationMetrics


def compute_homogenization_metrics(
    system: AbstractSystem,
    strings: Iterable[String]
) -> HomogenizationMetrics:
    """
    Compute all homogenization metrics for a system on a distribution.

    Paper (Section 4):
    "Homogenization minimizes deviance: E[∂_n] → 0, Var[∂_n] → 0, H(⟨Λ_n⟩) → 0"

    Args:
        system: The system to evaluate
        strings: Collection of strings (distribution sample)

    Returns:
        HomogenizationMetrics with all three measures
    """
    strings_list = list(strings)

    if not strings_list:
        return HomogenizationMetrics(
            expected_deviance=0.0,
            deviance_variance=0.0,
            core_entropy=0.0
        )

    # Compute core
    core_compliance = system.core(strings_list)

    # Compute orientations for each string
    deviances = []
    for s in strings_list:
        compliance = system.compliance(s)
        orientation = Orientation(
            compliance,
            core_compliance,
            difference_operator=system.difference_operator
        )
        deviances.append(orientation.deviance())

    # Compute metrics
    deviances_array = np.array(deviances)
    exp_dev = float(np.mean(deviances_array))
    dev_var = float(np.var(deviances_array))

    # Core entropy (only for vector systems)
    try:
        ent = core_entropy(core_compliance)
    except (AttributeError, TypeError):
        ent = 0.0

    return HomogenizationMetrics(
        expected_deviance=exp_dev,
        deviance_variance=dev_var,
        core_entropy=ent
    )
