"""
Continuation scorer.

Uses concatenation approach: tokenize prompt+continuation together,
run model once, sum log-probs for continuation tokens.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from exploration.common import ModelWrapper
from xenotechnics.common import String


class Scorer:
    """
    Score continuations using the concatenation approach.

    Tokenizes prompt+continuation together, runs model once,
    and computes log probability of continuation tokens.
    """

    def __init__(
        self,
        model_name: str = "",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        use_chat_template: bool = True,
        model: Optional[ModelWrapper] = None,
        lazy_load: bool = False,
        debug: bool = False,
    ):
        """Initialize scorer."""
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.use_chat_template = use_chat_template
        self.debug = debug

        if model is not None:
            self.model = model
        elif not lazy_load:
            self.model = ModelWrapper(model_name=model_name, device=device, dtype=dtype)
        else:
            self.model = None

    def _ensure_model_loaded(self) -> None:
        """Load model if not already loaded."""
        if self.model is None:
            self.model = ModelWrapper(
                model_name=self.model_name, device=self.device, dtype=self.dtype
            )

    def _has_chat_template(self) -> bool:
        """Check if tokenizer has a chat template."""
        return (
            hasattr(self.model.tokenizer, "chat_template")
            and self.model.tokenizer.chat_template is not None
        )

    def _prepare_texts(self, prompt_text: str, continuation: str) -> tuple[str, str]:
        """
        Prepare base and full texts for scoring.

        With chat template: formats as user message + assistant response.
        Without: simple concatenation.

        Returns:
            (base_text, full_text) - base_text is prompt only, full_text includes continuation
        """
        # Fall back to raw concatenation if no chat template available
        if not self.use_chat_template or not self._has_chat_template():
            if self.use_chat_template and self.debug:
                print("[DEBUG] No chat template available, using raw concatenation")
            return prompt_text, prompt_text + continuation

        # Build chat messages
        messages_base = [{"role": "user", "content": prompt_text}]
        messages_with_cont = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": continuation},
        ]

        # Apply chat template
        # add_generation_prompt=True adds the assistant turn prefix
        base_text = self.model.tokenizer.apply_chat_template(
            messages_base,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.model.tokenizer.apply_chat_template(
            messages_with_cont,
            tokenize=False,
            add_generation_prompt=False,
        )

        if self.debug:
            print(f"[DEBUG] Chat template base_text:\n{base_text}")
            print(f"[DEBUG] Chat template full_text:\n{full_text}")

        return base_text, full_text

    def score(
        self,
        prompt: Optional[String] = None,
        continuation: str = "",
        verbose: bool = False,
    ) -> dict:
        """
        Score a continuation given a prompt.

        Uses concatenation approach:
        1. Tokenize prompt + continuation together
        2. Run model once to get all logits
        3. Sum log-probs for continuation tokens

        With use_chat_template=True:
        - Prompt becomes user message
        - Continuation becomes assistant response
        - Scores P(response | formatted_conversation)

        Args:
            prompt: Prompt string (can be None for unconditional)
            continuation: Text to score
            verbose: Whether to print progress

        Returns:
            Dict with 'logprob', 'prob', 'n_tokens', 'per_token_logprobs'
        """
        self._ensure_model_loaded()

        prompt_text = prompt.to_text() if prompt else ""

        # Prepare texts (handles chat template if enabled)
        base_text, full_text = self._prepare_texts(prompt_text, continuation)

        # Tokenize base (prompt) and full text
        prompt_ids = (
            self.model.tokenizer.encode(base_text, add_special_tokens=False)
            if base_text
            else []
        )
        full_ids = self.model.tokenizer.encode(full_text, add_special_tokens=False)

        # Find where continuation starts by finding longest matching prefix
        # This handles tokenization merge at boundary (e.g., space + "queens" -> " queens")
        n_prompt = len(prompt_ids)
        match_len = 0
        for i in range(min(n_prompt, len(full_ids))):
            if full_ids[i] == prompt_ids[i]:
                match_len = i + 1
            else:
                break

        # Continuation tokens start after the matching prefix
        cont_start = match_len
        n_full = len(full_ids)
        n_cont = n_full - cont_start

        if self.debug:
            print(f"[DEBUG] prompt_ids ({n_prompt}): {prompt_ids}")
            print(f"[DEBUG] full_ids ({n_full}): {full_ids}")
            print(f"[DEBUG] match_len: {match_len}, cont_start: {cont_start}")
            print(f"[DEBUG] continuation tokens: {n_cont}")
            if match_len < n_prompt:
                print(
                    f"[DEBUG] tokenization merge detected at position {match_len}: "
                    f"prompt[{match_len}]={prompt_ids[match_len]} vs full[{match_len}]={full_ids[match_len]}"
                )

        # Handle edge case: empty continuation
        if not continuation:
            return {
                "logprob": 0.0,
                "prob": 1.0,
                "n_tokens": 0,
                "per_token_logprobs": [],
            }

        # If no continuation tokens after accounting for merge, that means
        # the continuation as-specified cannot be generated (tokenization issue)
        if n_cont <= 0:
            return {
                "logprob": float("-inf"),
                "prob": 0.0,
                "n_tokens": 0,
                "per_token_logprobs": [],
            }

        # Run model on full sequence
        input_ids = torch.tensor([full_ids], device=self.model.device)

        with torch.no_grad():
            outputs = self.model.model(input_ids)
            logits = outputs.logits[0]  # (seq_len, vocab_size)

        # Compute log-probs for each position
        log_probs = torch.log_softmax(logits, dim=-1)

        # Sum log-probs for continuation tokens
        # Token at position i is predicted by logits at position i-1
        total_logprob = 0.0
        per_token_logprobs = []

        for i in range(cont_start, n_full):
            token_id = full_ids[i]
            # Logits at position i-1 predict token at position i
            logit_pos = i - 1
            if logit_pos < 0:
                # First token has no conditioning (unconditional)
                continue

            token_logprob = log_probs[logit_pos, token_id].item()
            total_logprob += token_logprob

            token_str = self.model.tokenizer.decode([token_id])
            per_token_logprobs.append(
                {
                    "token": token_str,
                    "logprob": token_logprob,
                }
            )

            if verbose or self.debug:
                prob = np.exp(token_logprob)
                print(
                    f"  [DEBUG] step={i - cont_start:3d} "
                    f"token_id={token_id:6d} "
                    f"p={prob:.4e} "
                    f"logp={token_logprob:8.4f} "
                    f"{token_str!r}"
                )

        return {
            "logprob": total_logprob,
            "prob": np.exp(total_logprob),
            "n_tokens": len(per_token_logprobs),
            "per_token_logprobs": per_token_logprobs,
        }

    @classmethod
    def score_continuations(
        cls,
        model_name: str,
        prompt: str,
        continuations: list[str],
        verbose: bool = False,
        device: Optional[str] = None,
        use_chat_template: bool = False,
    ) -> dict:
        """
        Score multiple continuations for a prompt.

        Args:
            model_name: HuggingFace model name
            prompt: Prompt text
            continuations: List of continuations to score
            verbose: Whether to print progress
            device: Device to use
            use_chat_template: Whether to use chat template

        Returns:
            Dict with scores per continuation and rankings
        """
        scorer = cls(
            model_name=model_name,
            device=device,
            use_chat_template=use_chat_template,
        )

        prompt_string = String.from_text(prompt) if prompt else None
        scores = {}

        for cont in continuations:
            if verbose:
                print(f"\nScoring: '{cont}'")

            result = scorer.score(
                prompt=prompt_string,
                continuation=cont,
                verbose=verbose,
            )
            scores[cont] = result

        # Compute rankings
        ranked = sorted(scores.items(), key=lambda x: -x[1]["prob"])

        return {
            "prompt": prompt,
            "scores": scores,
            "ranking": [(cont, s["prob"]) for cont, s in ranked],
        }
