"""
System compliance representations.

Section 3.2: System compliance as first-class objects.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .system import AbstractSystem
    from .string import String


class AbstractSystemCompliance(ABC):
    """
    Abstract base class for system compliance Λ_n(x).

    Paper (Section 3.2, Equation 3):
    "The system compliance is a vector of compliances across particular
    structures: Λ_n(x) := (α_1(x), ..., α_n(x))"

    Attributes:
        system: The system that computed this compliance
        string: The string this compliance was computed for (None if core)
    """

    def __init__(
        self,
        system: AbstractSystem,
        string: Optional[String] = None
    ):
        """
        Initialize compliance.

        Args:
            system: The system that computed this compliance
            string: The string evaluated (None for core compliance)
        """
        self._system = system
        self.string = string

    @property
    def system(self) -> AbstractSystem:
        """Get the system that computed this compliance."""
        return self._system

    def __len__(self) -> int:
        """Number of structures."""
        return len(self._system)
