"""
Sampling exploration strategy.

Samples from distribution with temperature, top-k, and top-p.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import torch

from ..common import AbstractExplorer, ModelWrapper


class SamplingExplorer(AbstractExplorer):
    """
    Sampling exploration strategy.

    Samples tokens from the distribution with optional:
    - Temperature scaling
    - Top-k filtering
    - Top-p (nucleus) filtering
    """

    def _init_strategy_state(self, **kwargs) -> None:
        """
        Initialize sampling parameters.

        Args:
            temperature: Temperature for sampling (default: 1.0)
            top_k: Top-k filtering (default: None)
            top_p: Nucleus sampling threshold (default: None)
            seed: Random seed (default: None)
        """
        self.temperature = kwargs.get('temperature', 1.0)
        self.top_k = kwargs.get('top_k', None)
        self.top_p = kwargs.get('top_p', None)
        self.seed = kwargs.get('seed', None)

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

    def step_impl(
        self,
        logits: torch.Tensor,
        model: ModelWrapper,
        verbose: bool,
        **kwargs
    ) -> Optional[torch.Tensor]:
        """
        Sampling step: sample from distribution.

        Args:
            logits: Next token logits
            model: Model wrapper
            verbose: Whether to print progress
            **kwargs: Contains temperature, top_k, top_p

        Returns:
            Next token ID or None to stop
        """
        # Compute distribution
        probs = model.compute_distribution(logits[0])

        # Apply temperature scaling
        if self.temperature != 1.0:
            scaled_logits = logits[0] / self.temperature
            probs = torch.softmax(scaled_logits, dim=-1)

        # Convert to numpy for filtering
        probs_np = probs.cpu().numpy()

        # Apply top-k filtering
        if self.top_k is not None and self.top_k > 0:
            top_k_indices = np.argsort(probs_np)[-self.top_k:]
            filtered_probs = np.zeros_like(probs_np)
            filtered_probs[top_k_indices] = probs_np[top_k_indices]
            probs_np = filtered_probs / np.sum(filtered_probs)

        # Apply top-p (nucleus) filtering
        if self.top_p is not None and 0 < self.top_p < 1:
            sorted_indices = np.argsort(probs_np)[::-1]
            sorted_probs = probs_np[sorted_indices]
            cumsum_probs = np.cumsum(sorted_probs)
            cutoff_index = np.searchsorted(cumsum_probs, self.top_p) + 1
            top_p_indices = sorted_indices[:cutoff_index]
            filtered_probs = np.zeros_like(probs_np)
            filtered_probs[top_p_indices] = probs_np[top_p_indices]
            probs_np = filtered_probs / np.sum(filtered_probs)

        # Sample from distribution
        sampled_token_id = int(np.random.choice(len(probs_np), p=probs_np))
        next_token_id = torch.tensor([[sampled_token_id]], device=logits.device)

        token_str = model.tokenizer.decode([sampled_token_id])
        token_logprob = float(np.log(probs_np[sampled_token_id] + 1e-10))

        # Store data and build tree
        self._store_step_data(logits, next_token_id, token_str, token_logprob)

        # Print progress
        if verbose:
            sampled_prob = probs_np[sampled_token_id]
            # Compute entropy only for non-zero probabilities (0*log(0) = 0 by convention)
            mask = probs_np > 0
            entropy = float(-np.sum(probs_np[mask] * np.log(probs_np[mask])))
            print(
                f"Step {self.step_count - 1}: '{token_str}' "
                f"(p={sampled_prob:.4f}, H={entropy:.2f}, T={self.temperature:.2f})"
            )

        # Check if we should stop
        if self.step_count >= self.max_steps:
            return None

        return next_token_id
