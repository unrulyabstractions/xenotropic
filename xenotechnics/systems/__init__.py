"""System implementations."""

from .singleton_system import SingletonSystem
from .vector_system import VectorSystem
from .entropic import ExcessSystem, DeficitSystem
from .judge_vector_system import JudgeVectorSystem
from .judge_entropic_system import JudgeEntropicSystem
from .judge_generalized_system import JudgeGeneralizedSystem

__all__ = [
    'SingletonSystem',
    'VectorSystem',
    'ExcessSystem',
    'DeficitSystem',
    'JudgeVectorSystem',
    'JudgeEntropicSystem',
    'JudgeGeneralizedSystem',
]
