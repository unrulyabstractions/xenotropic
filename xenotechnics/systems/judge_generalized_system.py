"""
Judge-based generalized system implementation.

Implements generalized core from Appendix A using escort power mean formulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np

from xenotechnics.common import AbstractSystemCompliance, String

from .judge_vector_system import JudgeVectorSystem
from .vector_system import VectorSystemCompliance

if TYPE_CHECKING:
    from exploration.common import ModelRunner


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
        model_runner: Optional[ModelRunner] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        q: float = 1.0,
        r: float = 1.0,
    ):
        """
        Initialize judge generalized system.

        Args:
            questions: List of questions for LLM judges
            model_runner: Pre-loaded ModelRunner to share across structures
            model_name: Model name to load (required if model_runner not provided)
            device: Device to use (auto-detected if None)
            q: Power parameter for compliance values (default 1.0)
            r: Power parameter for probability weighting (default 1.0)
        """
        super().__init__(
            questions=questions,
            model_runner=model_runner,
            model_name=model_name,
            device=device,
        )

        self.q = q
        self.r = r

    def compute_core(
        self, trajectories: List[String], probabilities: np.ndarray
    ) -> AbstractSystemCompliance:
        """
        Compute generalized system core using escort power mean.

        From Appendix A:
        ⟨α_i^(q,r)⟩ = (Σ_y p(y)^r * α_i(y)^q / Σ_y p(y)^r)^(1/q)

        Args:
            trajectories: List of trajectory strings
            probabilities: Array of probabilities (must sum to 1)

        Returns:
            Core as VectorSystemCompliance
        """
        if not trajectories:
            raise ValueError("Cannot compute core from empty trajectory list")

        if len(trajectories) != len(probabilities):
            raise ValueError(
                "Number of trajectories must match number of probabilities"
            )

        # Compute escort distribution: p^(r)(y) = p(y)^r / Z
        escort_weights = np.power(probabilities, self.r)
        escort_weights /= escort_weights.sum()

        # Compute compliance for each trajectory
        compliance_vectors = np.array(
            [self.compliance(traj).to_array() for traj in trajectories]
        )  # Shape: (n_trajectories, n_structures)

        # Compute generalized core for each structure
        n_structures = len(self.structures)
        core_vector = np.zeros(n_structures)

        for i in range(n_structures):
            alpha_values = compliance_vectors[:, i]

            if self.q != 0:
                # Standard case: power mean
                alpha_q = np.power(alpha_values, self.q)
                expectation = np.sum(escort_weights * alpha_q)
                core_vector[i] = np.power(expectation, 1.0 / self.q)
            else:
                # Limit case q→0: geometric mean
                log_values = np.log(alpha_values + 1e-10)
                core_vector[i] = np.exp(np.sum(escort_weights * log_values))

        return VectorSystemCompliance(
            system=self,
            compliance_vector=core_vector,
            string=None,
        )

    def __repr__(self) -> str:
        return f"JudgeGeneralizedSystem({len(self.questions)} judges, q={self.q}, r={self.r})"
