"""
Tests for AbstractGenerator and generation infrastructure.

Tests for exploration/common/generator.py
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from xenotechnics.common import String
from xenotechnics.trees.tree import TreeNode


class TestGeneratorInterface:
    """Test generator interface without loading models."""

    def test_abstract_generator_import(self):
        """Test AbstractGenerator can be imported."""
        from exploration.common.generator import AbstractGenerator

        assert AbstractGenerator is not None

    def test_cannot_instantiate_abstract(self):
        """Test AbstractGenerator cannot be instantiated directly."""
        from exploration.common.generator import AbstractGenerator

        with pytest.raises(TypeError):
            AbstractGenerator("model_name")

    def test_model_wrapper_import(self):
        """Test ModelWrapper can be imported."""
        from exploration.common.model import ModelWrapper

        assert ModelWrapper is not None


class TestTreeBuilding:
    """Test tree building utilities."""

    def test_tree_node_creation(self):
        """Test creating tree nodes."""
        root = TreeNode(string=String.empty())
        assert root.string == String.empty()
        assert root.is_leaf()

    def test_add_child_builds_tree(self):
        """Test adding children builds tree structure."""
        root = TreeNode(string=String.empty())

        # Simulate adding tokens like a generator would
        child_a = root.add_child("The", logprob=-0.1, token_id=100)
        child_b = child_a.add_child(" ", logprob=-0.2, token_id=200)
        child_c = child_b.add_child("cat", logprob=-0.3, token_id=300)

        assert child_c.string.tokens == ("The", " ", "cat")
        assert child_c.depth() == 3
        assert not root.is_leaf()

    def test_set_distribution_on_node(self):
        """Test setting distribution on node."""
        node = TreeNode(string=String.empty())

        # Simulate distribution from logits
        probs = np.array([0.1, 0.2, 0.3, 0.4])
        node.set_distribution(probs=probs)

        assert node.next_token_distribution is not None
        np.testing.assert_array_almost_equal(node.next_token_distribution, probs)


class TestGeneratorMocking:
    """Test generator behavior with mocked dependencies."""

    @pytest.fixture
    def mock_model_wrapper(self):
        """Create mock ModelWrapper."""
        mock = MagicMock()
        mock.tokenizer.vocab_size = 100
        mock.tokenizer.eos_token_id = 0
        mock.tokenizer.decode.side_effect = lambda ids: "".join(
            chr(65 + i % 26) for i in ids
        )
        mock.tokenize_prompt.return_value = torch.tensor([[1, 2, 3]])
        mock.compute_distribution.return_value = torch.softmax(torch.randn(100), dim=0)
        return mock

    def test_tree_construction_flow(self, mock_model_wrapper):
        """Test the tree construction flow."""
        # Simulate what a generator does

        # 1. Create root
        root = TreeNode(string=String.empty())

        # 2. Add prompt tokens
        prompt_tokens = [("The", -0.1, 1), (" ", -0.2, 2), ("cat", -0.3, 3)]
        current = root
        for token, logprob, token_id in prompt_tokens:
            current = current.add_child(token, logprob, token_id)

        # 3. Add generated tokens with distributions
        probs = np.array([0.01] * 100)
        probs[50] = 0.9  # Peak at token 50
        current.set_distribution(probs=probs)

        child = current.add_child(" sat", logprob=np.log(0.8), token_id=50)
        child.mark_as_trajectory()

        # Verify tree structure
        assert len(root.get_trajectories()) == 1
        trajectory = root.get_trajectory_nodes()[0]
        assert trajectory.string.tokens == ("The", " ", "cat", " sat")


class TestDistributionHandling:
    """Test distribution handling in generation."""

    def test_distribution_normalization(self):
        """Test distributions are properly normalized."""
        node = TreeNode(string=String.empty())

        # Unnormalized-looking logits
        logits = np.array([1.0, 2.0, 3.0, 4.0])
        node.set_distribution(logits=logits)

        # Should be normalized after softmax
        assert np.sum(node.next_token_distribution) == pytest.approx(1.0)

    def test_distribution_temperature(self):
        """Test temperature affects distribution sharpness."""
        logits = np.array([0.0, 1.0, 2.0, 3.0])

        # Low temperature (sharp)
        node_low = TreeNode(string=String.empty())
        node_low.set_distribution(logits=logits, temperature=0.1)

        # High temperature (flat)
        node_high = TreeNode(string=String.empty())
        node_high.set_distribution(logits=logits, temperature=10.0)

        # Low temp should have higher peak
        assert np.max(node_low.next_token_distribution) > np.max(
            node_high.next_token_distribution
        )

    def test_child_logprobs_from_distribution(self):
        """Test setting child logprobs from distribution."""
        node = TreeNode(string=String.empty())

        probs = np.array([0.0, 0.3, 0.7, 0.0])  # Token 2 has 70%
        node.set_distribution(probs=probs)

        # Mock tokenizer
        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids: f"token_{ids[0]}"

        node.set_child_logprobs_from_distribution(
            distribution=probs, tokenizer=tokenizer, min_logprob=-20.0
        )

        # Should have logprobs for tokens with prob > 0
        assert "token_1" in node.child_logprobs
        assert "token_2" in node.child_logprobs
        assert node.child_logprobs["token_2"] == pytest.approx(np.log(0.7))
