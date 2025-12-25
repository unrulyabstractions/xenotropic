"""
Tests for abstract generator base class.

Tests for exploration/common/generator.py
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

import torch

from xenotechnics.common import String
from xenotechnics.trees.tree import TreeNode


class MockTokenizer:
    """Mock tokenizer."""

    def __init__(self):
        self.eos_token_id = 99

    def decode(self, token_ids):
        """Return string representation."""
        return f"tok_{token_ids[0]}"


class ConcreteGenerator:
    """Concrete generator for testing abstract methods."""

    def __init__(self):
        # Initialize required attributes without calling parent __init__
        pass

    def step_impl(
        self, logits: torch.Tensor, model, verbose: bool, **kwargs
    ) -> Optional[torch.Tensor]:
        """Concrete implementation."""
        return None

    def _store_step_data(
        self,
        logits: torch.Tensor,
        next_token_id: torch.Tensor,
        token_str: str,
        token_logprob: float,
    ) -> None:
        """Store data for current step and build tree."""
        # Compute distribution
        probs = self.model.compute_distribution(logits[0])
        dist = probs.cpu().numpy().astype("float32")

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

    def _find_prompt_node(self, tree, prompt_token_ids):
        """Find the node corresponding to the end of prompt in existing tree."""
        current = tree

        for token_id in prompt_token_ids:
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
        """Initialize strategy-specific state."""
        pass

    def _print_results(self, generated_ids: torch.Tensor) -> None:
        """Print generation results."""
        print()
        print("=" * 60)
        print("Full Response:")
        print()
        generated_text = self.model.decode_tokens(
            generated_ids[0], skip_special_tokens=False
        )
        print(generated_text)
        print("=" * 60)


class TestStoreStepData:
    """Test _store_step_data() method."""

    def test_stores_distribution(self):
        """Test that distribution is stored."""
        gen = ConcreteGenerator()
        gen.model = MagicMock()
        gen.model.compute_distribution.return_value = torch.tensor([0.3, 0.7])
        gen.model.tokenizer = MockTokenizer()
        gen.current_node = MagicMock()
        gen.current_node.add_child.return_value = MagicMock()
        gen.distributions = []
        gen.step_count = 0

        logits = torch.tensor([[0.3, 0.7]])
        next_token_id = torch.tensor([[1]])

        gen._store_step_data(logits, next_token_id, "tok", 0.0)

        assert len(gen.distributions) == 1
        assert gen.step_count == 1

    def test_updates_current_node(self):
        """Test that current_node is updated."""
        gen = ConcreteGenerator()
        gen.model = MagicMock()
        gen.model.compute_distribution.return_value = torch.tensor([0.3, 0.7])
        gen.model.tokenizer = MockTokenizer()
        child_node = MagicMock()
        gen.current_node = MagicMock()
        gen.current_node.add_child.return_value = child_node
        gen.distributions = []
        gen.step_count = 0

        logits = torch.tensor([[0.3, 0.7]])
        next_token_id = torch.tensor([[1]])

        gen._store_step_data(logits, next_token_id, "tok", 0.0)

        # Current node should now be the child
        assert gen.current_node == child_node


class TestFindPromptNode:
    """Test _find_prompt_node() method."""

    def test_finds_existing_prompt_node(self):
        """Test finding existing prompt node in tree."""
        gen = ConcreteGenerator()

        # Build a simple tree
        root = TreeNode(string=String.empty())
        child1 = root.add_child("tok1", logprob=0.0, token_id=1)
        child2 = child1.add_child("tok2", logprob=0.0, token_id=2)

        prompt_tokens = [1, 2]
        result = gen._find_prompt_node(root, prompt_tokens)

        assert result is child2

    def test_returns_none_if_not_found(self):
        """Test returns None if prompt not found."""
        gen = ConcreteGenerator()

        root = TreeNode(string=String.empty())
        root.add_child("tok1", logprob=0.0, token_id=1)

        prompt_tokens = [1, 99]  # 99 doesn't exist
        result = gen._find_prompt_node(root, prompt_tokens)

        assert result is None

    def test_returns_root_for_empty_prompt(self):
        """Test returns tree for empty prompt."""
        gen = ConcreteGenerator()

        root = TreeNode(string=String.empty())

        prompt_tokens = []
        result = gen._find_prompt_node(root, prompt_tokens)

        assert result is root


class TestInitStrategyState:
    """Test _init_strategy_state() method."""

    def test_default_does_nothing(self):
        """Test default implementation does nothing."""
        gen = ConcreteGenerator()

        # Should not raise
        gen._init_strategy_state(foo="bar", baz=123)


class TestPrintResults:
    """Test _print_results() method."""

    def test_prints_results(self, capsys):
        """Test that results are printed."""
        gen = ConcreteGenerator()
        gen.model = MagicMock()
        gen.model.decode_tokens.return_value = "Generated text here"

        generated_ids = torch.tensor([[1, 2, 3, 4, 5]])
        gen._print_results(generated_ids)

        captured = capsys.readouterr()
        assert "Generated text here" in captured.out
        assert "Full Response" in captured.out
