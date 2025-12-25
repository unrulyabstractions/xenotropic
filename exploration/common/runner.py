"""
Generation runner.

Handles the main generation loop with pluggable step functions.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .model import ModelWrapper

# Step function signature: (logits, model, generated_ids) -> next_token_id or None
StepFunction = Callable[
    [torch.Tensor, ModelWrapper, torch.Tensor], Optional[torch.Tensor]
]


class Runner:
    """
    Generation runner with pluggable step function.

    Handles the generation loop, calling a custom step function at each iteration.
    The step function determines the next token and when to stop.
    """

    def __init__(self, model: ModelWrapper, debug: bool = False):
        """
        Initialize runner.

        Args:
            model: Model wrapper for inference
            debug: Whether to print debug info
        """
        self.model = model
        self.debug = debug

    def run_generation(
        self, input_ids: torch.Tensor, run_step: StepFunction
    ) -> torch.Tensor:
        """
        Run generation with custom step function.

        The step function controls when to stop by returning None.

        Args:
            input_ids: Initial prompt token IDs
            run_step: Function (logits, model, generated_ids) -> next_id or None

        Returns:
            Generated token IDs (including prompt)
        """
        generated_ids = input_ids
        past_key_values = None

        if self.debug:
            print(f"[DEBUG] input_ids: {input_ids[0].tolist()}")

        while True:
            # Get next token logits
            logits, past_key_values = self.model.get_next_token_logits(
                input_ids=generated_ids, past_key_values=past_key_values
            )

            # Call custom step function with current generated_ids
            next_id = run_step(logits, self.model, generated_ids)

            # Check if generation should stop
            if next_id is None:
                break

            # Append next token
            generated_ids = torch.cat([generated_ids, next_id], dim=-1)

            # Check for EOS
            if (
                self.model.eos_token_id is not None
                and next_id.item() == self.model.eos_token_id
            ):
                break

        return generated_ids
