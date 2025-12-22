"""
Judge-based generalized system implementation.

Implements generalized core from Appendix A of the paper using
escort power mean formulation.
"""

from __future__ import annotations
from typing import List, Optional, Any

import numpy as np

from xenotechnics.common import AbstractSystemCompliance, String
from .judge_vector_system import JudgeVectorSystem
from .vector_system import VectorSystemCompliance


class JudgeGeneralizedSystem(JudgeVectorSystem):
    """
    Judge vector system with generalized core computation.

    Implements the escort power mean core from Appendix A:
    ⟨α_i^(q,r)⟩ = (E_{y~p^(r)}[α_i(y)^q])^(1/q)

    where p^(r) is the escort distribution.
    """

    def __init__(
        self,
        questions: List[str],
        model: Optional[Any] = None,
        model_name: Optional[str] = None,
        q: float = 1.0,
        r: float = 1.0,
    ):
        """
        Initialize judge generalized system.

        Args:
            questions: List of questions for LLM judges
            model: Pre-loaded model (optional)
            model_name: Model name to load (required if model not provided)
            q: Power parameter for compliance values (default 1.0)
            r: Power parameter for probability weighting (default 1.0)
        """
        super().__init__(
            questions=questions,
            model=model,
            model_name=model_name,
        )

        self.q = q
        self.r = r

    def compute_core(
        self,
        trajectories: List[String],
        probabilities: np.ndarray
    ) -> AbstractSystemCompliance:
        """
        Compute generalized system core using escort power mean.

        From Appendix A, Equation A.1 and A.3:
        ⟨α_i^(q,r)⟩ = (Σ_y p(y)^r * α_i(y)^q / Σ_y p(y)^r)^(1/q)

        This can be rewritten using escort distribution:
        p^(r)(y) = p(y)^r / Σ_y p(y)^r
        ⟨α_i^(q,r)⟩ = (E_{y~p^(r)}[α_i(y)^q])^(1/q)

        Args:
            trajectories: List of trajectory strings
            probabilities: Array of probabilities (must sum to 1)

        Returns:
            Core as VectorSystemCompliance
        """
        if not trajectories:
            raise ValueError("Cannot compute core from empty trajectory list")

        if len(trajectories) != len(probabilities):
            raise ValueError("Number of trajectories must match number of probabilities")

        # Compute escort distribution: p^(r)(y) = p(y)^r / Z
        escort_weights = np.power(probabilities, self.r)
        escort_weights /= escort_weights.sum()  # Normalize

        # Compute compliance for each trajectory
        compliance_vectors = []
        for traj in trajectories:
            compliance = self.compliance(traj)
            compliance_vectors.append(compliance.to_array())

        compliance_array = np.array(compliance_vectors)  # Shape: (n_trajectories, n_structures)

        # Compute generalized core for each structure
        # ⟨α_i^(q,r)⟩ = (E_{y~p^(r)}[α_i(y)^q])^(1/q)
        core_vector = np.zeros(len(self.structures))

        for i in range(len(self.structures)):
            # Get compliance values for structure i across all trajectories
            alpha_values = compliance_array[:, i]

            # Compute α^q
            alpha_q = np.power(alpha_values, self.q)

            # Compute escort-weighted expectation
            expectation = np.sum(escort_weights * alpha_q)

            # Take q-th root
            if self.q != 0:
                core_vector[i] = np.power(expectation, 1.0 / self.q)
            else:
                # Limit case: geometric mean
                log_values = np.log(alpha_values + 1e-10)
                core_vector[i] = np.exp(np.sum(escort_weights * log_values))

        return VectorSystemCompliance(
            system=self,
            compliance_vector=core_vector,
            string=None,  # Core has no associated string
        )

    def __repr__(self) -> str:
        return f"JudgeGeneralizedSystem({len(self.questions)} judges, q={self.q}, r={self.r})"
