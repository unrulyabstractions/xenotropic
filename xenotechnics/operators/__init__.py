"""
Operator implementations.

Operators are organized by their mathematical foundation:
- vector_operators: Standard Lp norms (L1, L2, L∞)
- entropic_operators: Information-theoretic measures (Rényi entropy)
- generalized_operators: Escort distributions and power means
"""

from .vector_operators import (
    VectorScoreOperator,
    VectorDifferenceOperator,
    L2ScoreOperator,
    L2SquaredScoreOperator,
    L1ScoreOperator,
    LinfScoreOperator,
    MeanScoreOperator,
    L2DifferenceOperator,
    L2SquaredDifferenceOperator,
    L1DifferenceOperator,
    LinfDifferenceOperator,
)

from .entropic_operators import (
    RelativeEntropy,
    NormalizedEntropy,
)

# TODO: Implement generalized_operators module
# from .generalized_operators import (
#     EscortPowerMeanOperator,
#     MaxExcessOperator,
#     MaxDeficitOperator,
# )

__all__ = [
    # Vector operator base classes
    'VectorScoreOperator',
    'VectorDifferenceOperator',
    # Vector operators
    'L2ScoreOperator',
    'L2SquaredScoreOperator',
    'L1ScoreOperator',
    'LinfScoreOperator',
    'MeanScoreOperator',
    'L2DifferenceOperator',
    'L2SquaredDifferenceOperator',
    'L1DifferenceOperator',
    'LinfDifferenceOperator',
    # Entropic operators
    'RelativeEntropy',
    'NormalizedEntropy',
    # Generalized operators (TODO: implement)
    # 'EscortPowerMeanOperator',
    # 'MaxExcessOperator',
    # 'MaxDeficitOperator',
]
