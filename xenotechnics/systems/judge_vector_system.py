"""
Judge-based vector system implementation.

Uses LLM judges to evaluate compliance across multiple criteria.
"""

from __future__ import annotations
from typing import List, Optional, Any, TYPE_CHECKING

from xenotechnics.structures import JudgeStructure
from .vector_system import VectorSystem

if TYPE_CHECKING:
    from xenotechnics.operators import VectorScoreOperator, VectorDifferenceOperator


class JudgeVectorSystem(VectorSystem):
    """
    Vector system using LLM judges.

    Each question becomes a JudgeStructure in the vector system.
    """

    def __init__(
        self,
        questions: List[str],
        model: Optional[Any] = None,
        model_name: Optional[str] = None,
        score_operator: Optional[VectorScoreOperator] = None,
        difference_operator: Optional[VectorDifferenceOperator] = None,
    ):
        """
        Initialize judge vector system.

        Args:
            questions: List of questions for LLM judges
            model: Pre-loaded model (optional, will be shared across structures)
            model_name: Model name to load (required if model not provided)
            score_operator: Score operator (defaults to L2SquaredScoreOperator)
            difference_operator: Difference operator (defaults to L2SquaredDifferenceOperator)
        """
        if not questions:
            raise ValueError("JudgeVectorSystem requires at least one question")

        if model is None and model_name is None:
            raise ValueError("Must provide either model or model_name")

        # Create JudgeStructure for each question
        structures = []
        for question in questions:
            judge = JudgeStructure(
                question=question,
                model=model,
                model_name=model_name if model is None else None,
            )
            structures.append(judge)

        # Initialize parent VectorSystem with JudgeStructures
        super().__init__(
            structures=structures,
            score_operator=score_operator,
            difference_operator=difference_operator,
        )

        self.questions = questions

    def __repr__(self) -> str:
        return f"JudgeVectorSystem({len(self.questions)} judges)"
