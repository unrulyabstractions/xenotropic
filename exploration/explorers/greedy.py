"""
Greedy exploration strategy.

Always selects the most probable token (argmax).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..common import AbstractExplorer, ModelWrapper


class GreedyExplorer(AbstractExplorer):
    """
    Greedy exploration strategy.

    Always selects argmax token at each step.
    Stops when max_new_tokens reached or EOS encountered.
    """

    def step_impl(
        self, logits: torch.Tensor, model: ModelWrapper, verbose: bool, **kwargs
    ) -> Optional[torch.Tensor]:
        """
        Greedy step: select argmax token.

        Args:
            logits: Next token logits
            model: Model wrapper
            verbose: Whether to print progress
            **kwargs: Unused

        Returns:
            Next token ID or None to stop
        """
        # Compute distribution
        probs = model.compute_distribution(logits[0])

        # Choose greedily (argmax)
        next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
        token_str = model.tokenizer.decode([next_token_id[0].item()])
        token_logprob = float(
            np.log(probs[next_token_id[0].item()].cpu().item() + 1e-10)
        )

        # Store data and build tree
        self._store_step_data(logits, next_token_id, token_str, token_logprob)

        # Print progress
        if verbose:
            sampled_prob = probs[next_token_id[0]].cpu().item()
            # Compute entropy only for non-zero probabilities (0*log(0) = 0 by convention)
            mask = probs > 0
            entropy = float(
                -torch.sum(probs[mask] * torch.log(probs[mask])).cpu().item()
            )
            print(
                f"Step {self.step_count - 1}: '{token_str}' "
                f"(p={sampled_prob:.4f}, H={entropy:.2f})"
            )

        # Check if we should stop
        if self.step_count >= self.max_steps:
            return None

        return next_token_id
