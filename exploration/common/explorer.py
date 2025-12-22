"""
Abstract explorer base class.

Defines interface for exploration strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch

from xenotechnics.common import String
from xenotechnics.trees.tree import TreeNode

from .model import ModelWrapper
from .runner import Runner


class AbstractExplorer(ABC):
    """
    Abstract base class for exploration strategies.

    Provides common infrastructure for:
    - Model and runner management
    - State management during generation
    - Tree building with proper probabilities

    Subclasses implement specific search strategies via step_impl().
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        use_chat_template: bool = True,
    ):
        """
        Initialize explorer.

        Args:
            model_name: HuggingFace model name
            device: Device to use (auto-detected if None)
            dtype: Data type (auto-detected if None)
            use_chat_template: Whether to use chat template for prompts
        """
        self.model = ModelWrapper(model_name=model_name, device=device, dtype=dtype)
        self.runner = Runner(self.model)
        self.use_chat_template = use_chat_template

        # State for current generation
        self.distributions = []
        self.current_node = None
        self.root_node = None
        self.step_count = 0
        self.prompt_token_count = 0  # Number of prompt tokens (edges from root)

    def run(
        self,
        prompt: Optional[String] = None,
        max_new_tokens: int = 100,
        verbose: bool = True,
        existing_tree: Optional[TreeNode] = None,
        **kwargs,
    ) -> TreeNode:
        """
        Run exploration and return TreeNode.

        Args:
            prompt: Prompt string (None for empty prompt)
            max_new_tokens: Maximum tokens to generate
            verbose: Whether to print progress
            existing_tree: Optional existing tree to reuse (skips prompt rebuild)
            **kwargs: Strategy-specific parameters

        Returns:
            TreeNode with trajectory (root of tree)
        """
        # Convert prompt to text
        if prompt is None:
            prompt_text = ""
        else:
            prompt_text = prompt.to_text()

        if verbose:
            print("=" * 60)
            print(f"Prompt: {prompt_text}")
            print("=" * 60)
            print()

        # Tokenize prompt
        input_ids = self.model.tokenize_prompt(
            prompt_text, use_chat_template=self.use_chat_template
        )

        # Initialize state (reuse existing tree if provided)
        self._init_generation_state(input_ids, max_new_tokens, existing_tree, **kwargs)

        # Define step function
        def step_fn(
            logits: torch.Tensor, model: ModelWrapper
        ) -> Optional[torch.Tensor]:
            return self.step_impl(logits, model, verbose, **kwargs)

        # Run generation
        generated_ids = self.runner.run_generation(
            input_ids=input_ids, run_step=step_fn
        )

        # Mark the final node as a complete trajectory
        self.current_node.mark_as_trajectory()

        if verbose:
            self._print_results(generated_ids)

        return self.root_node

    @abstractmethod
    def step_impl(
        self, logits: torch.Tensor, model: ModelWrapper, verbose: bool, **kwargs
    ) -> Optional[torch.Tensor]:
        """
        Implement step function for this exploration strategy.

        Args:
            logits: Next token logits
            model: Model wrapper
            verbose: Whether to print progress
            **kwargs: Strategy-specific parameters

        Returns:
            Next token ID or None to stop
        """
        pass

    def _init_generation_state(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        existing_tree: Optional[TreeNode] = None,
        **kwargs,
    ) -> None:
        """
        Initialize generation state and compute prompt probabilities.

        Args:
            input_ids: Prompt token IDs
            max_new_tokens: Maximum tokens to generate
            existing_tree: Optional existing tree to reuse
            **kwargs: Strategy-specific initialization parameters
        """
        self.distributions = []
        self.step_count = 0
        self.max_steps = max_new_tokens

        prompt_token_ids = input_ids[0].tolist()

        # If existing tree provided, find the prompt node and reuse
        if existing_tree is not None:
            prompt_node = self._find_prompt_node(existing_tree, prompt_token_ids)
            if prompt_node is not None:
                self.root_node = existing_tree
                self.current_node = prompt_node
                self.prompt_token_count = len(prompt_token_ids)
                self._init_strategy_state(**kwargs)
                return

        # Create new root node with empty string
        self.root_node = TreeNode(string=String.empty())
        self.current_node = self.root_node

        # Build prompt chain with proper probabilities
        for i, token_id in enumerate(prompt_token_ids):
            token_str = self.model.tokenizer.decode([token_id])

            # For first token, we can't compute probability (no prior context)
            # Just add with logprob 0.0
            if i == 0:
                child = self.current_node.add_child(
                    token=token_str, logprob=0.0, token_id=token_id
                )
                self.current_node = child
                continue

            # Get prefix up to this point
            prefix_ids = input_ids[:, :i]

            # Get logits for next token given prefix
            logits, _ = self.model.get_next_token_logits(
                input_ids=prefix_ids, past_key_values=None
            )

            # Compute distribution
            probs = self.model.compute_distribution(logits[0])
            dist = probs.cpu().numpy().astype(np.float32)

            # Set distribution on current node
            self.current_node.set_distribution(probs=dist)

            # Set child_logprobs from distribution
            self.current_node.set_child_logprobs_from_distribution(
                distribution=dist, tokenizer=self.model.tokenizer
            )

            # Get token string and its log probability
            token_logprob = float(np.log(probs[token_id].item() + 1e-10))

            # Add child node
            child = self.current_node.add_child(
                token=token_str, logprob=token_logprob, token_id=token_id
            )
            self.current_node = child

        # Store prompt token count (number of edges from root to prompt node)
        self.prompt_token_count = len(prompt_token_ids)

        # Allow subclasses to add custom initialization
        self._init_strategy_state(**kwargs)

    def _find_prompt_node(
        self, tree: TreeNode, prompt_token_ids: list
    ) -> Optional[TreeNode]:
        """
        Find the node corresponding to the end of prompt in existing tree.

        Traverses tree following prompt token IDs.

        Args:
            tree: Root of existing tree
            prompt_token_ids: List of prompt token IDs

        Returns:
            TreeNode at end of prompt, or None if not found
        """
        current = tree

        for token_id in prompt_token_ids:
            # Look for child with matching token_id
            found = False
            for child in current.children.values():
                if child.token_id == token_id:
                    current = child
                    found = True
                    break

            if not found:
                return None

        return current

    def _init_strategy_state(self, **kwargs) -> None:
        """
        Initialize strategy-specific state.

        Subclasses can override to add custom initialization.

        Args:
            **kwargs: Strategy-specific parameters
        """
        pass

    def _print_results(self, generated_ids: torch.Tensor) -> None:
        """
        Print generation results.

        Args:
            generated_ids: All generated token IDs
        """
        print()
        print("=" * 60)
        print("Full Response:")
        print()
        generated_text = self.model.decode_tokens(
            generated_ids[0], skip_special_tokens=False
        )
        print(generated_text)
        print("=" * 60)

    def _store_step_data(
        self,
        logits: torch.Tensor,
        next_token_id: torch.Tensor,
        token_str: str,
        token_logprob: float,
    ) -> None:
        """
        Store data for current step and build tree.

        Args:
            logits: Next token logits
            next_token_id: Selected token ID
            token_str: Selected token string
            token_logprob: Log probability of selected token
        """
        # Compute distribution
        probs = self.model.compute_distribution(logits[0])
        dist = probs.cpu().numpy().astype(np.float32)

        # Store distribution
        self.distributions.append(dist)

        # Set distribution on current node
        self.current_node.set_distribution(probs=dist)

        # Set child_logprobs from distribution
        self.current_node.set_child_logprobs_from_distribution(
            distribution=dist, tokenizer=self.model.tokenizer
        )

        # Add child node
        child = self.current_node.add_child(
            token=token_str, logprob=token_logprob, token_id=next_token_id[0].item()
        )

        # Move to child
        self.current_node = child

        # Increment step
        self.step_count += 1
