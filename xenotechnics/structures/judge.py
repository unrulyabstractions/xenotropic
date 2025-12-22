"""
Judge structure using LLM evaluation.

Uses a language model to evaluate strings against a question.
"""

from __future__ import annotations
from typing import Optional, Any
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..common import AbstractStructure, String


class JudgeStructure(AbstractStructure):
    """
    Structure that uses an LLM to judge string compliance.

    Given a question, prompts an LLM to score the string from 0 to 1.
    """

    def __init__(
        self,
        question: str,
        model: Optional[Any] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize judge structure.

        Args:
            question: Question to ask the LLM about the string
            model: Pre-loaded model wrapper (optional)
            model_name: Model name to load (required if model not provided)
            device: Device to use (auto-detected if None)
            dtype: Data type (auto-detected if None)
        """
        super().__init__(
            name=f"Judge: {question[:50]}...",
            description=f"LLM judge evaluating: {question}"
        )

        self.question = question

        if model is not None:
            self.model = model
            self.tokenizer = model.tokenizer
            self.device = model.device
        elif model_name is not None:
            self._init_model(model_name, device, dtype)
        else:
            raise ValueError("Must provide either model or model_name")

    def _init_model(
        self,
        model_name: str,
        device: Optional[str],
        dtype: Optional[torch.dtype]
    ) -> None:
        """Initialize model and tokenizer."""
        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        # Auto-detect dtype
        if dtype is None:
            if device in ["mps", "cuda"]:
                dtype = torch.float16
            else:
                dtype = torch.float32

        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device,
        )
        self.model.eval()

    def compliance(self, string: String) -> float:
        """
        Compute compliance by prompting LLM to score the string.

        Args:
            string: String to evaluate

        Returns:
            Compliance score in [0, 1]
        """
        # Get text from string
        text = string.to_text()

        # Create prompt
        prompt = self._format_prompt(text)

        # Get LLM response
        score = self._query_model(prompt)

        return score

    def _format_prompt(self, text: str) -> str:
        """
        Format prompt for LLM judge.

        Args:
            text: Text to evaluate

        Returns:
            Formatted prompt
        """
        prompt = f"""Evaluate the following text based on this question:

Question: {self.question}

Text: {text}

Provide a score from 0.0 to 1.0, where:
- 0.0 means the text completely fails the criterion
- 1.0 means the text perfectly satisfies the criterion

Respond with ONLY a number between 0.0 and 1.0, nothing else.

Score:"""

        return prompt

    def _query_model(self, prompt: str) -> float:
        """
        Query the model and parse score.

        Args:
            prompt: Formatted prompt

        Returns:
            Parsed score in [0, 1]
        """
        # Tokenize
        if hasattr(self, 'tokenizer') and self.tokenizer.chat_template is not None:
            input_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            input_ids = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
            )

        input_ids = input_ids.to(self.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response (only new tokens)
        response = self.tokenizer.decode(
            output_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )

        # Parse score
        score = self._parse_score(response)

        return score

    def _parse_score(self, response: str) -> float:
        """
        Parse score from model response.

        Args:
            response: Model response text

        Returns:
            Parsed score in [0, 1], defaults to 0.5 if unparseable
        """
        # Try to find a decimal number
        match = re.search(r'(\d+\.?\d*)', response.strip())

        if match:
            try:
                score = float(match.group(1))
                # Clamp to [0, 1]
                score = max(0.0, min(1.0, score))
                return score
            except ValueError:
                pass

        # Default to 0.5 if unparseable
        return 0.5
