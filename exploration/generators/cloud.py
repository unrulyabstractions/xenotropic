"""
Cloud-based generators using HuggingFace Inference API.

These generators don't load models locally - they call the HuggingFace
Inference API for generation. Useful for large models that don't fit
in local memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from xenotechnics.common import String
from xenotechnics.trees.tree import TreeNode


@dataclass
class CloudScore:
    """Score result from cloud inference."""

    text: str
    prob: float
    logprob: float
    n_tokens: int = 1


def _get_inference_client():
    """Get HuggingFace InferenceClient (lazy import)."""
    from huggingface_hub import InferenceClient

    return InferenceClient()


class CloudGreedyGenerator:
    """
    Cloud-based greedy generator using HuggingFace Inference API.

    Does not load models locally - calls cloud API for generation.
    """

    def __init__(self, model_name: str):
        """
        Initialize cloud generator.

        Args:
            model_name: HuggingFace model name (must be available via Inference API)
        """
        self.model_name = model_name
        self._client = None

    @property
    def client(self):
        """Lazy-load inference client."""
        if self._client is None:
            self._client = _get_inference_client()
        return self._client

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        verbose: bool = False,
    ) -> CloudScore:
        """
        Generate text greedily via cloud API.

        Args:
            prompt: Prompt text
            max_new_tokens: Maximum tokens to generate
            verbose: Whether to print progress

        Returns:
            CloudScore with generated text and probabilities
        """
        if verbose:
            print(f"[CLOUD] Generating with {self.model_name}...", flush=True)

        try:
            response = self.client.text_generation(
                prompt,
                model=self.model_name,
                max_new_tokens=max_new_tokens,
                details=True,
            )

            gen_text = (
                response.generated_text
                if hasattr(response, "generated_text")
                else str(response)
            )

            # Sum logprobs if available
            total_lp = 0.0
            n_tokens = 0
            if hasattr(response, "details") and response.details:
                tokens = response.details.tokens or []
                n_tokens = len(tokens)
                total_lp = sum(
                    t.logprob for t in tokens if hasattr(t, "logprob") and t.logprob
                )

            if verbose:
                print(f"  Generated {n_tokens} tokens", flush=True)

            return CloudScore(
                text=gen_text,
                prob=math.exp(total_lp) if total_lp else 0.0,
                logprob=total_lp,
                n_tokens=n_tokens,
            )

        except Exception as e:
            if verbose:
                print(f"  Error: {e}", flush=True)
            return CloudScore(text="", prob=0.0, logprob=float("-inf"), n_tokens=0)

    def greedy_next(self, prompt: str, verbose: bool = False) -> CloudScore:
        """
        Get most probable next token via cloud API.

        Args:
            prompt: Prompt text
            verbose: Whether to print progress

        Returns:
            CloudScore with next token
        """
        if verbose:
            print(f"[CLOUD] Getting next token from {self.model_name}...", flush=True)

        try:
            response = self.client.text_generation(
                prompt,
                model=self.model_name,
                max_new_tokens=1,
                details=True,
                decoder_input_details=True,
            )

            if hasattr(response, "details") and response.details:
                tokens = response.details.tokens
                if tokens:
                    token = tokens[0]
                    logprob = token.logprob if hasattr(token, "logprob") else 0.0
                    return CloudScore(
                        text=token.text,
                        prob=math.exp(logprob),
                        logprob=logprob,
                    )

            # Fallback
            gen_text = (
                response.generated_text
                if hasattr(response, "generated_text")
                else str(response)
            )
            return CloudScore(text=gen_text, prob=0.0, logprob=float("-inf"))

        except Exception as e:
            if verbose:
                print(f"  Error: {e}", flush=True)
            return CloudScore(text="", prob=0.0, logprob=float("-inf"))

    def run(
        self,
        prompt: Optional[String] = None,
        max_new_tokens: int = 100,
        verbose: bool = True,
        **kwargs,
    ) -> TreeNode:
        """
        Run generation and return a simple TreeNode.

        Note: Cloud API doesn't provide full distribution, so tree
        will have limited probability information.

        Args:
            prompt: Prompt string
            max_new_tokens: Maximum tokens to generate
            verbose: Whether to print progress

        Returns:
            TreeNode with generated trajectory
        """
        prompt_text = prompt.to_text() if prompt else ""
        result = self.generate(prompt_text, max_new_tokens, verbose)

        # Build simple tree
        root = TreeNode(string=String.empty())
        current = root

        # Add prompt (without probabilities - we don't have them from cloud)
        if prompt and prompt.tokens:
            for token in prompt.tokens:
                child = current.add_child(token=token, logprob=0.0)
                current = child

        # Add generated tokens (we only have total logprob, not per-token)
        if result.text:
            # Split by character as approximation (we don't have tokenization)
            avg_logprob = result.logprob / max(result.n_tokens, 1)
            for char in result.text:
                child = current.add_child(token=char, logprob=avg_logprob)
                current = child

        current.mark_as_trajectory()
        return root


class CloudScorerGenerator:
    """
    Cloud-based scorer using HuggingFace Inference API.

    Scores continuations via cloud API. Note that cloud API may not
    provide exact logprobs for specific continuations - this uses
    approximations.
    """

    def __init__(self, model_name: str):
        """
        Initialize cloud scorer.

        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self._client = None

    @property
    def client(self):
        """Lazy-load inference client."""
        if self._client is None:
            self._client = _get_inference_client()
        return self._client

    def score(
        self,
        prompt: str,
        continuation: str,
        verbose: bool = False,
    ) -> dict:
        """
        Score a continuation given a prompt.

        Note: Cloud API approximates this by generating and checking
        if the generation matches the continuation.

        Args:
            prompt: Prompt text
            continuation: Continuation to score
            verbose: Whether to print progress

        Returns:
            Dict with 'logprob', 'prob', 'n_tokens'
        """
        if verbose:
            print(
                f"[CLOUD] Scoring '{continuation}' with {self.model_name}...",
                flush=True,
            )

        try:
            # Generate with the prompt and request logprobs
            response = self.client.text_generation(
                prompt,
                model=self.model_name,
                max_new_tokens=len(continuation.split()) + 10,
                details=True,
                decoder_input_details=True,
            )

            # Try to match continuation tokens
            if hasattr(response, "details") and response.details:
                tokens = response.details.tokens or []

                # Build generated text and find where continuation starts
                gen_text = ""
                logprobs = []
                for tok in tokens:
                    gen_text += tok.text
                    if hasattr(tok, "logprob") and tok.logprob:
                        logprobs.append(tok.logprob)

                # Check if continuation is prefix of generated text
                if gen_text.startswith(continuation):
                    # Sum logprobs for continuation tokens
                    n_cont_tokens = min(len(logprobs), len(continuation.split()) + 5)
                    total_lp = sum(logprobs[:n_cont_tokens])
                    return {
                        "logprob": total_lp,
                        "prob": math.exp(total_lp),
                        "n_tokens": n_cont_tokens,
                        "matched": True,
                    }

                # Approximation: use available logprobs
                if logprobs:
                    total_lp = sum(logprobs)
                    return {
                        "logprob": total_lp,
                        "prob": math.exp(total_lp),
                        "n_tokens": len(logprobs),
                        "matched": False,
                        "generated": gen_text[:50],
                    }

            return {
                "logprob": float("-inf"),
                "prob": 0.0,
                "n_tokens": 0,
                "matched": False,
            }

        except Exception as e:
            if verbose:
                print(f"  Error: {e}", flush=True)
            return {
                "logprob": float("-inf"),
                "prob": 0.0,
                "n_tokens": 0,
                "error": str(e),
            }

    @classmethod
    def score_continuations(
        cls,
        model_name: str,
        prompt: str,
        continuations: list[str],
        verbose: bool = False,
    ) -> dict:
        """
        Score multiple continuations for a prompt.

        Args:
            model_name: HuggingFace model name
            prompt: Prompt text
            continuations: List of continuations to score
            verbose: Whether to print progress

        Returns:
            Dict with scores per continuation and rankings
        """
        scorer = cls(model_name)
        scores = {}

        for cont in continuations:
            result = scorer.score(prompt, cont, verbose)
            scores[cont] = result

        # Compute rankings
        ranked = sorted(scores.items(), key=lambda x: -x[1]["prob"])

        return {
            "prompt": prompt,
            "model": model_name,
            "scores": scores,
            "ranking": [(cont, s["prob"]) for cont, s in ranked],
        }
