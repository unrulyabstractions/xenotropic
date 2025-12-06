"""
Judge structure using LLM evaluation.

Uses ModelRunner for simple text generation to evaluate strings.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

from ..common import AbstractStructure, String

if TYPE_CHECKING:
    from exploration.common import ModelRunner


class JudgeStructure(AbstractStructure):
    """
    Structure that uses an LLM to judge string compliance.

    Given a question, prompts an LLM to score the string from 0 to 1.
    Uses ModelRunner for efficient generation.
    """

    PROMPT_TEMPLATE = """You are a precise evaluator. Answer the following question about the given text with ONLY a single number between 0 and 1, where 0 means "not at all" and 1 means "completely/definitely". Do not include any other text, explanation, or punctuation - just the number.

Text: "{text}"

Question: {question}

Your numeric answer (0-1):"""

    def __init__(
        self,
        question: str,
        model_runner: Optional[ModelRunner] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize judge structure.

        Args:
            question: Question to ask the LLM about the string
            model_runner: Pre-loaded ModelRunner to share across structures
            model_name: Model name to load (required if model_runner not provided)
            device: Device to use (auto-detected if None)
        """
        super().__init__(
            name=f"Judge: {question[:50]}{'...' if len(question) > 50 else ''}",
            description=f"LLM judge evaluating: {question}",
        )

        self.question = question
        self._model_runner = model_runner
        self._model_name = model_name
        self._device = device

    @property
    def model_runner(self) -> ModelRunner:
        """Lazy-load model runner."""
        if self._model_runner is None:
            from exploration.common import ModelRunner

            if self._model_name is None:
                raise ValueError("Must provide model_runner or model_name")
            self._model_runner = ModelRunner(
                model_name=self._model_name,
                device=self._device,
            )
        return self._model_runner

    def compliance(self, string: String) -> float:
        """
        Compute compliance by prompting LLM to score the string.

        Args:
            string: String to evaluate

        Returns:
            Compliance score in [0, 1]
        """
        text = string.to_text()
        return self.judge(text)[0]

    def judge(self, text: str) -> tuple[float, str]:
        """
        Judge a text string directly.

        Args:
            text: Text to evaluate

        Returns:
            Tuple of (score, raw_response)
        """
        prompt = self.PROMPT_TEMPLATE.format(question=self.question, text=text)

        # Generate response (short, greedy)
        response = self.model_runner.generate(
            prompt,
            max_new_tokens=20,
            temperature=0.0,
            apply_chat_template=True,
        )

        score = self._parse_score(response)
        return score, response

    def _parse_score(self, response: str) -> float:
        """
        Parse score from model response.

        Args:
            response: Model response text

        Returns:
            Parsed score in [0, 1], defaults to 0.5 if unparseable
        """
        response = response.strip()

        # Strategy 1: Direct parsing
        try:
            score = float(response)
            if 0 <= score <= 1:
                return score
            # Only treat as percentage if it looks like one (integer values > 1)
            if 1 < score <= 100 and score == int(score):
                return score / 100
            # Clamp other values
            return max(0.0, min(1.0, score))
        except ValueError:
            pass

        # Strategy 2: Find first decimal number in response
        match = re.search(r"(\d+\.?\d*)", response)
        if match:
            try:
                score = float(match.group(1))
                if 0 <= score <= 1:
                    return score
                # Only treat as percentage if it looks like one (integer values > 1)
                if 1 < score <= 100 and score == int(score):
                    return score / 100
                # Clamp other values
                return max(0.0, min(1.0, score))
            except ValueError:
                pass

        # Strategy 3: Common text patterns
        response_lower = response.lower()
        # Use word boundaries to avoid matching "no" in "unknown"
        words = set(response_lower.split())
        if words & {"no", "not", "zero", "none", "never"}:
            return 0.0
        if any(
            w in response_lower
            for w in ["yes", "completely", "definitely", "one", "full"]
        ):
            return 1.0
        if "half" in response_lower or "middle" in response_lower:
            return 0.5

        # Default to 0.5 if unparseable
        return 0.5
