"""
Xenotechnics: Structure-aware Diversity Pursuit for LLMs

Implementation of the theoretical framework from:
"Structure-aware Diversity Pursuit as AI Safety strategy against Homogenization"
"""

__version__ = "0.1.0"

# Core abstractions
from .common import (
    String,
    AbstractStructure,
    FunctionalStructure,
    CompositeStructure,
    AbstractSystem,
    AbstractScoreOperator,
    AbstractDifferenceOperator,
    Orientation,
)

# Structure implementations
from .structures import (
    ClassifierStructure,
    MultiClassifierStructure,
    SimilarityStructure,
    MultiReferenceSimilarityStructure,
    GrammarStructure,
    ValidWordsStructure,
    SentenceStructureStructure,
    POSPatternStructure,
    ReadabilityStructure,
)

# Operator implementations
from .operators import (
    L2ScoreOperator,
    L1ScoreOperator,
    LinfScoreOperator,
    MeanScoreOperator,
    L2DifferenceOperator,
    L1DifferenceOperator,
    LinfDifferenceOperator,
    # TODO: Implement missing operators
    # RenyiExcessOperator,
    # RenyiDeficitOperator,
    # EscortPowerMeanOperator,
    # MaxExcessOperator,
    # MaxDeficitOperator,
)

# System implementations
from .systems import (
    VectorSystem,
    SingletonSystem,
    ExcessSystem,
    DeficitSystem,
)

# Tree implementations
from .trees import TreeNode

# Xeno-reproduction
from .xenoreproduction import (
    HomogenizationMetrics,
    InterventionScores,
    compute_homogenization_metrics,
    score_diversity,
    score_intervention,
)

# Dynamics
from .dynamics import (
    AbstractDynamics,
    LinearDynamics,
    DynamicsState,
)

__all__ = [
    # Core abstractions
    "String",
    "AbstractStructure",
    "FunctionalStructure",
    "CompositeStructure",
    "AbstractSystem",
    "AbstractScoreOperator",
    "AbstractDifferenceOperator",
    "Orientation",
    # Structures
    "ClassifierStructure",
    "MultiClassifierStructure",
    "SimilarityStructure",
    "MultiReferenceSimilarityStructure",
    "GrammarStructure",
    "ValidWordsStructure",
    "SentenceStructureStructure",
    "POSPatternStructure",
    "ReadabilityStructure",
    # Operators
    "L2ScoreOperator",
    "L1ScoreOperator",
    "LinfScoreOperator",
    "MeanScoreOperator",
    "L2DifferenceOperator",
    "L1DifferenceOperator",
    "LinfDifferenceOperator",
    # TODO: Implement missing operators
    # "RenyiExcessOperator",
    # "RenyiDeficitOperator",
    # "EscortPowerMeanOperator",
    # "MaxExcessOperator",
    # "MaxDeficitOperator",
    # Systems
    "VectorSystem",
    "SingletonSystem",
    "ExcessSystem",
    "DeficitSystem",
    # Trees
    "TreeNode",
    # Xeno-reproduction
    "HomogenizationMetrics",
    "InterventionScores",
    "compute_homogenization_metrics",
    "score_diversity",
    "score_intervention",
    # Dynamics
    "AbstractDynamics",
    "LinearDynamics",
    "DynamicsState",
]
