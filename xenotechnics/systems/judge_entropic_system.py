"""
Judge-based entropic system implementation.

Uses LLM judges with Rényi entropy-based operators.
Supports excess and deficit modes (Appendix A.2, Equations A.7 and A.8).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Optional

from xenotechnics.common import AbstractDifferenceOperator, AbstractSystemCompliance

from .judge_vector_system import JudgeVectorSystem

if TYPE_CHECKING:
    from exploration.common import ModelRunner


class ExcessDifferenceOperator(AbstractDifferenceOperator):
    """
    Wrapper for excess deviance mode.

    Excess deviance (Appendix A.2, Equation A.7):
    ∂_q^+(y, x_p) = e^(H_q(Λ_n(y) || ⟨Λ_n⟩(x_p)))

    Measures effective over-representation/over-compliance.
    """

    def __init__(self, q: float = 2.0):
        from xenotechnics.operators import RelativeEntropy

        self.q = q
        self.relative_entropy = RelativeEntropy(q=q)

    def __call__(
        self,
        compliance1: AbstractSystemCompliance,
        compliance2: AbstractSystemCompliance,
    ) -> float:
        """Compute excess deviance: H_q(compliance1 || compliance2)."""
        return self.relative_entropy(compliance1, compliance2)


class DeficitDifferenceOperator(AbstractDifferenceOperator):
    """
    Wrapper for deficit deviance mode.

    Deficit deviance (Appendix A.2, Equation A.8):
    ∂_q^-(y, x_p) = e^(H_q(⟨Λ_n⟩(x_p) || Λ_n(y)))

    Measures effective under-representation/under-compliance.
    """

    def __init__(self, q: float = 2.0):
        from xenotechnics.operators import RelativeEntropy

        self.q = q
        self.relative_entropy = RelativeEntropy(q=q)

    def __call__(
        self,
        compliance1: AbstractSystemCompliance,
        compliance2: AbstractSystemCompliance,
    ) -> float:
        """Compute deficit deviance: H_q(compliance2 || compliance1)."""
        return self.relative_entropy(compliance2, compliance1)


class JudgeEntropicSystem(JudgeVectorSystem):
    """
    Judge vector system using Rényi entropy-based operators.

    Uses RelativeEntropy (Rényi divergence) for difference measurement and
    NormalizedEntropy for score computation.

    Supports two modes:
    - "excess": Measures over-compliance H_q(Λ || ⟨Λ⟩)
    - "deficit": Measures under-compliance H_q(⟨Λ⟩ || Λ)
    """

    def __init__(
        self,
        questions: List[str],
        model_runner: Optional[ModelRunner] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        q: float = 2.0,
        mode: Literal["excess", "deficit"] = "excess",
    ):
        """
        Initialize judge entropic system.

        Args:
            questions: List of questions for LLM judges
            model_runner: Pre-loaded ModelRunner to share across structures
            model_name: Model name to load (required if model_runner not provided)
            device: Device to use (auto-detected if None)
            q: Order of Rényi entropy (default 2.0)
            mode: "excess" for over-compliance or "deficit" for under-compliance
        """
        from xenotechnics.operators import NormalizedEntropy

        self.q = q
        self.mode = mode

        # Create appropriate difference operator based on mode
        if mode == "excess":
            difference_op = ExcessDifferenceOperator(q=q)
        elif mode == "deficit":
            difference_op = DeficitDifferenceOperator(q=q)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'excess' or 'deficit'")

        super().__init__(
            questions=questions,
            model_runner=model_runner,
            model_name=model_name,
            device=device,
            score_operator=NormalizedEntropy(q=q),
            difference_operator=difference_op,
        )

    def __repr__(self) -> str:
        return (
            f"JudgeEntropicSystem("
            f"{len(self.questions)} judges, "
            f"q={self.q}, "
            f"mode={self.mode})"
        )
