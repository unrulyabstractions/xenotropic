"""
Abstract base classes and interfaces.

This module contains the fundamental abstractions of the framework:
- Strings (token sequences)
- Structure interface
- System interface and compliance representations
- Operator interfaces
- Orientation representations
"""

from .string import String
from .structure import AbstractStructure, FunctionalStructure, CompositeStructure
from .system import AbstractSystem
from .operator import AbstractScoreOperator, AbstractDifferenceOperator
from .compliance import AbstractSystemCompliance
from .orientation import Orientation

__all__ = [
    "String",
    "AbstractStructure",
    "FunctionalStructure",
    "CompositeStructure",
    "AbstractSystem",
    "AbstractScoreOperator",
    "AbstractDifferenceOperator",
    "AbstractSystemCompliance",
    "Orientation",
]
