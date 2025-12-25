"""
Abstract base classes and interfaces.

This module contains the fundamental abstractions of the framework:
- Strings (token sequences)
- Structure interface
- System interface and compliance representations
- Operator interfaces
- Orientation representations
"""

from .compliance import AbstractSystemCompliance
from .operator import AbstractDifferenceOperator, AbstractScoreOperator
from .orientation import Orientation
from .schema_utils import SchemaClass
from .string import String
from .structure import AbstractStructure, CompositeStructure, FunctionalStructure
from .system import AbstractSystem

__all__ = [
    "AbstractDifferenceOperator",
    "AbstractScoreOperator",
    "AbstractStructure",
    "AbstractSystem",
    "AbstractSystemCompliance",
    "CompositeStructure",
    "FunctionalStructure",
    "Orientation",
    "SchemaClass",
    "String",
]
