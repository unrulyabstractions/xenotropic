"""
Abstract base classes for trajectory dynamics.

Section 3.5: Dynamics framework.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, TYPE_CHECKING

import numpy as np

from xenotechnics.common import AbstractSystem, String

if TYPE_CHECKING:
    from xenotechnics.common import String


@dataclass
class DynamicsState:
    """
    State at step k during trajectory generation.

    Paper (Section 3.5, Equation 10):
    "Given a trajectory y = x_T, for k ∈ {0, 1, ..., T}, we define states:

    x_ϕ_k = ⟨Λ_n⟩(x_k)   # Expected compliance of continuations from x_k
    y_ϕ_k = θ_n(x_k|⊥)    # Orientation of current path from root
    z_ϕ_k = θ_n(y|x_k)    # Orientation of trajectory from current position"

    Attributes:
        step: Step index k
        current_string: String x_k at this step
        x_phi: Expected compliance of continuations
        y_phi: Orientation from root to current position
        z_phi: Orientation from current position to trajectory end
    """
    step: int
    current_string: String
    x_phi: np.ndarray
    y_phi: np.ndarray
    z_phi: np.ndarray

    def __repr__(self) -> str:
        return (
            f"DynamicsState(step={self.step}, "
            f"||x_ϕ||={np.linalg.norm(self.x_phi):.3f}, "
            f"||y_ϕ||={np.linalg.norm(self.y_phi):.3f}, "
            f"||z_ϕ||={np.linalg.norm(self.z_phi):.3f})"
        )


class AbstractDynamics(ABC):
    """
    Abstract base class for trajectory dynamics tracking.

    Paper (Section 3.5):
    "As a string is being completed, the set of possible trajectories is
    narrowed so the system core and orientations change."

    Tracks the discrete-time dynamics:
    (x_ϕ₀, y_ϕ₀, z_ϕ₀) → ... → (x_ϕ_T, y_ϕ_T, z_ϕ_T)
    """

    def __init__(self, system: AbstractSystem):
        """
        Initialize dynamics tracker.

        Args:
            system: The system to track
        """
        self.system = system
        self.states: List[DynamicsState] = []

    @abstractmethod
    def compute_state(
        self,
        step: int,
        current_string: String,
        trajectory: String,
        root_core: np.ndarray,
        continuation_core: np.ndarray
    ) -> DynamicsState:
        """
        Compute the dynamics state at step k.

        Args:
            step: Step index k
            current_string: Current string x_k
            trajectory: Full trajectory y
            root_core: Core from root ⟨Λ_n⟩(⊥)
            continuation_core: Core from current position ⟨Λ_n⟩(x_k)

        Returns:
            DynamicsState at step k
        """
        pass

    def track_trajectory(
        self,
        trajectory: String,
        root_core: np.ndarray,
        continuation_cores: List[np.ndarray]
    ) -> None:
        """
        Track the dynamics of generating a trajectory.

        Args:
            trajectory: The complete trajectory
            root_core: Core from root ⟨Λ_n⟩(⊥)
            continuation_cores: List of ⟨Λ_n⟩(x_k) for each step k
        """
        self.states.clear()

        for k in range(len(trajectory)):
            current_string = String(trajectory.tokens[:k + 1])

            if k < len(continuation_cores):
                cont_core = continuation_cores[k]
            else:
                # At the end, use trajectory compliance
                traj_compliance = self.system.compliance(trajectory)
                cont_core = traj_compliance.to_array()

            state = self.compute_state(
                step=k,
                current_string=current_string,
                trajectory=trajectory,
                root_core=root_core,
                continuation_core=cont_core
            )

            self.states.append(state)

    def get_evolution(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the evolution of all three state components over time.

        Returns:
            Tuple of (x_phis, y_phis, z_phis), each of shape (T, n)
            where T is number of steps and n is system dimension

        Paper (Section 3.5):
        "which form a discrete-time dynamics"
        """
        if not self.states:
            return np.array([]), np.array([]), np.array([])

        x_phis = np.array([s.x_phi for s in self.states])
        y_phis = np.array([s.y_phi for s in self.states])
        z_phis = np.array([s.z_phi for s in self.states])

        return x_phis, y_phis, z_phis

    def __len__(self) -> int:
        """Number of tracked states."""
        return len(self.states)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self.states)} states)"
