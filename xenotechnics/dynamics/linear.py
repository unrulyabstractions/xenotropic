"""
Linear (vector-based) dynamics implementation.

Section 3.5: Default dynamics for vector systems.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from xenotechnics.common import AbstractSystem, String

from .base import AbstractDynamics, DynamicsState

if TYPE_CHECKING:
    from xenotechnics.common import String


class LinearDynamics(AbstractDynamics):
    """
    Linear dynamics for vector-based systems.

    Computes dynamics states using simple vector arithmetic:
    - x_ϕ_k = continuation_core (given)
    - y_ϕ_k = Λ_n(x_k) - root_core
    - z_ϕ_k = Λ_n(y) - continuation_core

    This is the default and most common dynamics implementation.
    """

    def compute_state(
        self,
        step: int,
        current_string: String,
        trajectory: String,
        root_core: np.ndarray,
        continuation_core: np.ndarray
    ) -> DynamicsState:
        """
        Compute linear dynamics state.

        Args:
            step: Step index k
            current_string: Current string x_k
            trajectory: Full trajectory y
            root_core: Core from root ⟨Λ_n⟩(⊥)
            continuation_core: Core from current position ⟨Λ_n⟩(x_k)

        Returns:
            DynamicsState with linear computations
        """
        # x_ϕ_k = ⟨Λ_n⟩(x_k)
        x_phi = continuation_core

        # y_ϕ_k = θ_n(x_k|⊥) = Λ_n(x_k) - ⟨Λ_n⟩(⊥)
        current_compliance = self.system.compliance(current_string)
        current_array = current_compliance.to_array()
        y_phi = current_array - root_core

        # z_ϕ_k = θ_n(y|x_k) = Λ_n(y) - ⟨Λ_n⟩(x_k)
        trajectory_compliance = self.system.compliance(trajectory)
        trajectory_array = trajectory_compliance.to_array()
        z_phi = trajectory_array - continuation_core

        return DynamicsState(
            step=step,
            current_string=current_string,
            x_phi=x_phi,
            y_phi=y_phi,
            z_phi=z_phi
        )
