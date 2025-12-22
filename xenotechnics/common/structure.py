"""
Abstract structure interface and common structure implementations.

Section 3.2: Structure-awareness
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List
import numpy as np

from .string import String


class AbstractStructure(ABC):
    """
    Abstract base class for structures.

    Paper (Section 3.2):
    "We define structure as the specification of a type of organization
    among the tokens of a string."

    The degree of structure compliance α_i(x) ∈ [0,1] measures how well
    a string conforms to this organization.
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    @abstractmethod
    def compliance(self, string: String) -> float:
        """
        Compute structure compliance α_i(x) ∈ [0, 1].

        Paper (Section 3.2, Equation 2):
        "For a string x ∈ Str, the degree of structure compliance is
        α_i(x). Ideal compliance corresponds to α_i(x) = 1, and no
        compliance corresponds to α_i(x) = 0."

        Args:
            string: The string to evaluate

        Returns:
            Compliance score in [0, 1]
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class FunctionalStructure(AbstractStructure):
    """
    Structure defined by custom compliance function.

    Allows creating structures from arbitrary functions without subclassing.
    """

    def __init__(
        self,
        compliance_fn: Callable[[String], float],
        name: str = "custom",
        description: str = "Custom compliance function"
    ):
        """
        Initialize FunctionalStructure.

        Args:
            compliance_fn: Function that takes a String and returns compliance in [0, 1]
            name: Structure name
            description: Structure description
        """
        super().__init__(name, description)
        self.compliance_fn = compliance_fn

    def compliance(self, string: String) -> float:
        result = self.compliance_fn(string)
        return max(0.0, min(1.0, float(result)))


class CompositeStructure(AbstractStructure):
    """
    Combine multiple structures with aggregation function.

    Allows building complex structures from simpler ones.
    """

    def __init__(
        self,
        structures: List[AbstractStructure],
        combiner: str = "mean",
        name: str = "composite"
    ):
        """
        Initialize CompositeStructure.

        Args:
            structures: List of structures to combine
            combiner: How to combine scores ("mean", "min", "max", "product")
            name: Structure name
        """
        super().__init__(name, f"Composite of {len(structures)} structures")
        self.structures = structures
        self.combiner = combiner

    def compliance(self, string: String) -> float:
        scores = [s.compliance(string) for s in self.structures]

        if self.combiner == "mean":
            return float(np.mean(scores))
        elif self.combiner == "min":
            return float(np.min(scores))
        elif self.combiner == "max":
            return float(np.max(scores))
        elif self.combiner == "product":
            return float(np.prod(scores))
        else:
            raise ValueError(f"Unknown combiner: {self.combiner}")
