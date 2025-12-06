"""
Entropy-based operators using Rényi divergence.

Implements operators based on Rényi entropy (Appendix A.2) for
measuring differences and concentration in compliance distributions.
"""

from __future__ import annotations

import numpy as np
from xenotechnics.common import AbstractScoreOperator, AbstractDifferenceOperator
from xenotechnics.systems.vector_system import VectorSystemCompliance


class RelativeEntropy(AbstractDifferenceOperator):
    """
    Rényi divergence operator (Appendix A.2).

    Computes Rényi divergence of order q:
    H_q(p||r) = 1/(q-1) * log(Σ_i p_i^q * r_i^(1-q))

    The compliance vectors are normalized to probability distributions
    before computing the divergence.

    Special cases:
    - q=1: Shannon entropy (KL divergence)
    - q=2: Collision entropy
    """

    def __init__(self, q: float = 2.0):
        """
        Initialize Rényi divergence operator.

        Args:
            q: Order of Rényi entropy (default 2.0)
        """
        self.q = q

    def __call__(
        self,
        compliance1: VectorSystemCompliance,
        compliance2: VectorSystemCompliance
    ) -> float:
        """
        Compute Rényi divergence H_q(p||r).

        Args:
            compliance1: First compliance (p distribution)
            compliance2: Second compliance (r distribution)

        Returns:
            Rényi divergence
        """
        vec1 = compliance1.to_array()
        vec2 = compliance2.to_array()

        # Normalize to probability distributions
        p = vec1 / (vec1.sum() + 1e-10)
        r = vec2 / (vec2.sum() + 1e-10)

        # Clip to avoid numerical issues
        p = np.clip(p, 1e-10, 1.0)
        r = np.clip(r, 1e-10, 1.0)

        if self.q == 1.0:
            # Shannon entropy (KL divergence) - limit as q→1
            return float(np.sum(p * np.log(p / r)))
        else:
            # Rényi divergence: H_q(p||r) = 1/(q-1) * log(Σ p^q * r^(1-q))
            sum_term = np.sum(np.power(p, self.q) * np.power(r, 1 - self.q))
            return float(np.log(sum_term + 1e-10) / (self.q - 1))


class NormalizedEntropy(AbstractScoreOperator):
    """
    Normalized Rényi entropy score operator.

    Computes Rényi entropy relative to uniform distribution, normalized by log(n):

    H_q,normalized(Λ) = H_q(normalized(Λ) || uniform) / log(n)

    This measures how concentrated the compliance distribution is relative
    to the maximally uniform (high entropy) distribution.

    A value of 0 means the distribution is uniform (maximum entropy).
    Higher values indicate more concentration (lower entropy).
    """

    def __init__(self, q: float = 2.0):
        """
        Initialize normalized entropy operator.

        Args:
            q: Order of Rényi entropy (default 2.0)
        """
        self.q = q
        self.relative_entropy = RelativeEntropy(q=q)

    def __call__(self, compliance: VectorSystemCompliance) -> float:
        """
        Compute normalized Rényi entropy score.

        Args:
            compliance: Compliance vector to evaluate

        Returns:
            Normalized entropy score
        """
        vec = compliance.to_array()
        n = len(vec)

        # Normalize to probability distribution
        p = vec / (vec.sum() + 1e-10)

        # Create uniform distribution
        uniform = np.ones(n) / n

        # Create compliance objects for relative entropy
        p_compliance = VectorSystemCompliance(
            system=compliance.system,
            compliance_vector=p,
            string=None
        )
        uniform_compliance = VectorSystemCompliance(
            system=compliance.system,
            compliance_vector=uniform,
            string=None
        )

        # Compute H_q(p || uniform)
        renyi_div = self.relative_entropy(p_compliance, uniform_compliance)

        # Normalize by log(n)
        if n > 1:
            normalized_entropy = renyi_div / np.log(n)
        else:
            normalized_entropy = 0.0

        return float(normalized_entropy)
