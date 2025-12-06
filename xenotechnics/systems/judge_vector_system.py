"""
Judge-based vector system implementation.

Uses LLM judges to evaluate compliance across multiple criteria.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from xenotechnics.structures import JudgeStructure

from .vector_system import VectorSystem

if TYPE_CHECKING:
    from exploration.common import ModelRunner
    from xenotechnics.operators import VectorDifferenceOperator, VectorScoreOperator


class JudgeVectorSystem(VectorSystem):
    """
    Vector system using LLM judges.

    Each question becomes a JudgeStructure in the vector system.
    All structures share a single ModelRunner for efficiency.
    """

    def __init__(
        self,
        questions: List[str],
        model_runner: Optional[ModelRunner] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        score_operator: Optional[VectorScoreOperator] = None,
        difference_operator: Optional[VectorDifferenceOperator] = None,
    ):
        """
        Initialize judge vector system.

        Args:
            questions: List of questions for LLM judges
            model_runner: Pre-loaded ModelRunner to share across structures
            model_name: Model name to load (required if model_runner not provided)
            device: Device to use (auto-detected if None)
            score_operator: Score operator (defaults to L2SquaredScoreOperator)
            difference_operator: Difference operator (defaults to L2SquaredDifferenceOperator)
        """
        if not questions:
            raise ValueError("JudgeVectorSystem requires at least one question")

        if model_runner is None and model_name is None:
            raise ValueError("Must provide model_runner or model_name")

        # Create shared model runner if not provided
        if model_runner is None:
            from exploration.common import ModelRunner

            model_runner = ModelRunner(model_name=model_name, device=device)

        # Create JudgeStructure for each question, sharing the model runner
        structures = [
            JudgeStructure(question=q, model_runner=model_runner) for q in questions
        ]

        super().__init__(
            structures=structures,
            score_operator=score_operator,
            difference_operator=difference_operator,
        )

        self.questions = questions
        self._model_runner = model_runner

    @property
    def model_runner(self) -> ModelRunner:
        """Get the shared model runner."""
        return self._model_runner

    def __repr__(self) -> str:
        return f"JudgeVectorSystem({len(self.questions)} judges)"
