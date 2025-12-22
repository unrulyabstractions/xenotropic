"""
Concrete structure implementations.

Structures are organized by their implementation approach:
- classifier: PyTorch classifier-based structures
- similarity: Embedding similarity-based structures
- grammar: Grammar and pattern-based structures

Note: FunctionalStructure and CompositeStructure are in xenotechnics.common
"""

from .classifier import (
    ClassifierStructure,
    MultiClassifierStructure,
)
from .similarity import (
    SimilarityStructure,
    MultiReferenceSimilarityStructure,
)
from .grammar import (
    GrammarStructure,
    ValidWordsStructure,
    SentenceStructureStructure,
    POSPatternStructure,
    ReadabilityStructure,
)
from .judge import JudgeStructure

__all__ = [
    # Classifier structures
    'ClassifierStructure',
    'MultiClassifierStructure',
    # Similarity structures
    'SimilarityStructure',
    'MultiReferenceSimilarityStructure',
    # Grammar structures
    'GrammarStructure',
    'ValidWordsStructure',
    'SentenceStructureStructure',
    'POSPatternStructure',
    'ReadabilityStructure',
    # Judge structures
    'JudgeStructure',
]
