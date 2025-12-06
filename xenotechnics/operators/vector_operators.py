"""
Vector-based operators using Lp norms.

Standard operators from the main paper (Sections 3.2, 3.3).
These operators use standard vector norms (L1, L2, L∞) to measure
compliance and differences.
"""

from __future__ import annotations

import numpy as np
from xenotechnics.common import AbstractScoreOperator, AbstractDifferenceOperator
from xenotechnics.systems.vector_system import VectorSystemCompliance


class VectorScoreOperator(AbstractScoreOperator):
    """
    Base class for score operators that work with VectorSystemCompliance.
    """

    def __call__(self, compliance: VectorSystemCompliance) -> float:
        raise NotImplementedError


class VectorDifferenceOperator(AbstractDifferenceOperator):
    """
    Base class for difference operators that work with VectorSystemCompliance.
    """

    def __call__(self, compliance1: VectorSystemCompliance, compliance2: VectorSystemCompliance) -> float:
        raise NotImplementedError


class L2ScoreOperator(VectorScoreOperator):
    """
    L2 norm for system scores.

    Computes ||Λ_n(x)||_2 normalized by √n.
    """

    def __call__(self, compliance: VectorSystemCompliance) -> float:
        compliance_vector = compliance.to_array()
        return float(np.linalg.norm(compliance_vector, ord=2) / np.sqrt(len(compliance)))


class L2SquaredScoreOperator(VectorScoreOperator):
    """
    Squared L2 norm for system scores.

    Computes ||Λ_n(x)||_2^2 normalized by n.
    """

    def __call__(self, compliance: VectorSystemCompliance) -> float:
        compliance_vector = compliance.to_array()
        return float(np.dot(compliance_vector, compliance_vector) / len(compliance))


class L1ScoreOperator(VectorScoreOperator):
    """
    L1 norm for system scores.

    Computes ||Λ_n(x)||_1 normalized by n.
    """

    def __call__(self, compliance: VectorSystemCompliance) -> float:
        compliance_vector = compliance.to_array()
        return float(np.linalg.norm(compliance_vector, ord=1) / len(compliance))


class LinfScoreOperator(VectorScoreOperator):
    """
    L-infinity norm for system scores.

    Computes max_i |α_i(x)|.
    """

    def __call__(self, compliance: VectorSystemCompliance) -> float:
        compliance_vector = compliance.to_array()
        return float(np.linalg.norm(compliance_vector, ord=np.inf))


class MeanScoreOperator(VectorScoreOperator):
    """
    Mean aggregation for system scores.

    Computes (1/n) Σ α_i(x).
    """

    def __call__(self, compliance: VectorSystemCompliance) -> float:
        compliance_vector = compliance.to_array()
        return float(np.mean(compliance_vector))


class L2DifferenceOperator(VectorDifferenceOperator):
    """
    L2 norm for system differences.

    Computes ||Λ_n(x) - Λ_m(x)||_2 for two compliances.
    """

    def __call__(self, compliance1: VectorSystemCompliance, compliance2: VectorSystemCompliance) -> float:
        vec1 = compliance1.to_array()
        vec2 = compliance2.to_array()
        diff = vec1 - vec2
        n = min(len(compliance1), len(compliance2))
        return float(np.linalg.norm(diff, ord=2) / np.sqrt(n))


class L2SquaredDifferenceOperator(VectorDifferenceOperator):
    """
    Squared L2 norm for system differences.

    Computes ||Λ_n(x) - Λ_m(x)||_2^2 normalized by n.
    """

    def __call__(self, compliance1: VectorSystemCompliance, compliance2: VectorSystemCompliance) -> float:
        vec1 = compliance1.to_array()
        vec2 = compliance2.to_array()
        diff = vec1 - vec2
        n = min(len(compliance1), len(compliance2))
        return float(np.dot(diff, diff) / n)


class L1DifferenceOperator(VectorDifferenceOperator):
    """
    L1 norm for system differences.

    Computes ||Λ_n(x) - Λ_m(x)||_1 for two compliances.
    """

    def __call__(self, compliance1: VectorSystemCompliance, compliance2: VectorSystemCompliance) -> float:
        vec1 = compliance1.to_array()
        vec2 = compliance2.to_array()
        diff = vec1 - vec2
        n = min(len(compliance1), len(compliance2))
        return float(np.linalg.norm(diff, ord=1) / n)


class LinfDifferenceOperator(VectorDifferenceOperator):
    """
    L-infinity norm for system differences.

    Computes max_i |α_i^(1)(x) - α_i^(2)(x)|.
    """

    def __call__(self, compliance1: VectorSystemCompliance, compliance2: VectorSystemCompliance) -> float:
        vec1 = compliance1.to_array()
        vec2 = compliance2.to_array()
        diff = vec1 - vec2
        return float(np.linalg.norm(diff, ord=np.inf))
