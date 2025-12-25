"""
Judge structure using LLM evaluation.

Uses a language model to evaluate strings against a question.
"""

from __future__ import annotations

import re
from typing import Optional

from ..common import AbstractStructure, String


class JudgeStructure(AbstractStructure):
    """
    Structure that uses an LLM to judge string compliance.

    Given a question, prompts an LLM to score the string from 0 to 1.
    Uses GreedyGenerator or CloudGreedyGenerator internally.
    """

    PROMPT_TEMPLATE = """You are a precise evaluator. Answer the following question about the given text with ONLY a single number between 0 and 1, where 0 means "not at all" and 1 means "completely/definitely". Do not include any other text, explanation, or punctuation - just the number.

Text: "{text}"

Question: {question}

Your numeric answer (0-1):"""

    def __init__(
        self,
        question: str,
        model_name: str,
        device: Optional[str] = None,
        use_cloud: bool = False,
        use_chat_template: bool = True,
        isolate: bool = False,
    ):
        """
        Initialize judge structure.

        Args:
            question: Question to ask the LLM about the string
            model_name: Model name to load
            device: Device to use (auto-detected if None)
            use_cloud: If True, use CloudGreedyGenerator for HuggingFace Inference API
            use_chat_template: Whether to use chat template for prompts
            isolate: If True, run generation in subprocess for memory isolation
        """
        super().__init__(
            name=f"Judge: {question[:50]}...",
            description=f"LLM judge evaluating: {question}",
        )

        self.question = question
        self.model_name = model_name
        self.device = device
        self.use_cloud = use_cloud
        self.use_chat_template = use_chat_template
        self.isolate = isolate
        self._generator = None

    @property
    def generator(self):
        """Lazy-load generator."""
        if self._generator is None:
            if self.use_cloud:
                from exploration.generators import CloudGreedyGenerator

                self._generator = CloudGreedyGenerator(self.model_name)
            else:
                from exploration.generators import GreedyGenerator

                self._generator = GreedyGenerator(
                    model_name=self.model_name,
                    device=self.device,
                    use_chat_template=self.use_chat_template,
                    lazy_load=self.isolate,
                )
        return self._generator

    def compliance(self, string: String) -> float:
        """
        Compute compliance by prompting LLM to score the string.

        Args:
            string: String to evaluate

        Returns:
            Compliance score in [0, 1]
        """
        text = string.to_text()
        prompt = self._format_prompt(text)

        if self.use_cloud:
            response = self._query_cloud(prompt)
        else:
            response = self._query_local(prompt)

        return self._parse_score(response)

    def judge(self, text: str) -> tuple[float, str]:
        """
        Judge a text string directly.

        Convenience method that accepts text instead of String object.

        Args:
            text: Text to evaluate

        Returns:
            Tuple of (score, raw_response)
        """
        prompt = self._format_prompt(text)

        if self.use_cloud:
            response = self._query_cloud(prompt)
        else:
            response = self._query_local(prompt)

        score = self._parse_score(response)
        return score, response

    def _format_prompt(self, text: str) -> str:
        """Format prompt for LLM judge."""
        return self.PROMPT_TEMPLATE.format(question=self.question, text=text)

    def _query_local(self, prompt: str) -> str:
        """Query using local GreedyGenerator."""
        prompt_string = String.from_text(prompt)

        tree = self.generator.run(
            prompt=prompt_string,
            max_new_tokens=20,
            verbose=False,
            isolate=self.isolate,
        )

        # Get trajectory text (continuation only)
        trajectories = tree.get_trajectory_nodes()
        if trajectories:
            traj = trajectories[0]
            full_text = traj.string.to_text()

            # Extract assistant response from chat template if present
            # Look for common chat template patterns
            if "<|im_start|>assistant" in full_text:
                # Qwen/ChatML style: <|im_start|>assistant\n...<|im_end|>
                match = re.search(
                    r"<\|im_start\|>assistant\n?(.*?)(?:<\|im_end\|>|$)",
                    full_text,
                    re.DOTALL,
                )
                response = match.group(1).strip() if match else ""
            elif "[/INST]" in full_text:
                # Llama style: [/INST] ...
                match = re.search(r"\[/INST\]\s*(.*?)(?:</s>|$)", full_text, re.DOTALL)
                response = match.group(1).strip() if match else ""
            elif "<|assistant|>" in full_text:
                # Phi style: <|assistant|>...
                match = re.search(
                    r"<\|assistant\|>\s*(.*?)(?:<\|end\|>|$)", full_text, re.DOTALL
                )
                response = match.group(1).strip() if match else ""
            else:
                # Fallback: extract continuation after prompt
                response = full_text[len(prompt) :].strip()
        else:
            response = ""

        return response

    def _query_cloud(self, prompt: str) -> str:
        """Query using CloudGreedyGenerator."""
        try:
            result = self.generator.generate(prompt, max_new_tokens=20, verbose=False)
            return result.text.strip()
        except Exception as e:
            return f"[Error: {e}]"

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
            if 1 < score <= 100:
                return score / 100
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
                if 1 < score <= 100:
                    return score / 100
                return max(0.0, min(1.0, score))
            except ValueError:
                pass

        # Strategy 3: Common text patterns
        response_lower = response.lower()
        if any(w in response_lower for w in ["no", "not", "zero", "none"]):
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
