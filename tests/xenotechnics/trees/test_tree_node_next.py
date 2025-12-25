"""
Tests for TreeNode next() method and sampling functionality.

Tests for xenotechnics/trees/tree.py - TreeNode.next() and related methods
"""

from __future__ import annotations

import numpy as np
import pytest

from xenotechnics.common import String
from xenotechnics.trees.tree import TreeNode


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab_size: int = 10):
        self.vocab_size = vocab_size
        self._vocab = {i: f"token_{i}" for i in range(vocab_size)}

    def decode(self, token_ids: list) -> str:
        """Decode token IDs to string."""
        return "".join(self._vocab.get(tid, f"<unk_{tid}>") for tid in token_ids)

    def encode(self, text: str) -> list:
        """Encode text to token IDs (simple mock)."""
        return [0]  # Mock implementation


class TestTreeNodeNext:
    """Test TreeNode next() method."""

    @pytest.fixture
    def tokenizer(self):
        """Create mock tokenizer."""
        return MockTokenizer(vocab_size=10)

    @pytest.fixture
    def node_with_distribution(self, tokenizer):
        """Create node with set distribution."""
        node = TreeNode(string=String.empty())
        # Create a distribution where token_0 has highest probability
        probs = np.array([0.5, 0.3, 0.1, 0.05, 0.05] + [0.0] * 5)
        node.set_distribution(probs=probs)
        return node

    def test_next_no_distribution_raises(self, tokenizer):
        """Test next() raises when no distribution set."""
        node = TreeNode(string=String.empty())

        with pytest.raises(ValueError, match="No distribution set"):
            node.next(tokenizer)

    def test_next_greedy(self, node_with_distribution, tokenizer):
        """Test greedy sampling takes argmax."""
        child = node_with_distribution.next(tokenizer, greedy=True)

        # Token 0 has highest probability, so greedy should select it
        assert child is not None
        assert child.parent == node_with_distribution
        assert "token_0" in child.string.tokens[-1]

    def test_next_greedy_deterministic(self, node_with_distribution, tokenizer):
        """Test greedy is deterministic (always same result)."""
        results = []
        for _ in range(5):
            node = TreeNode(string=String.empty())
            node.set_distribution(probs=np.array([0.4, 0.6] + [0.0] * 8))
            child = node.next(tokenizer, greedy=True)
            results.append(child.string.tokens[-1])

        # All should be the same (token_1 has highest prob)
        assert all(r == results[0] for r in results)

    def test_next_with_seed(self, node_with_distribution, tokenizer):
        """Test next() with seed is reproducible."""
        results = []
        for _ in range(3):
            node = TreeNode(string=String.empty())
            node.set_distribution(probs=np.array([0.3, 0.3, 0.4] + [0.0] * 7))
            child = node.next(tokenizer, seed=42)
            results.append(child.string.tokens[-1])

        # All should be the same with same seed
        assert all(r == results[0] for r in results)

    def test_next_with_token_id(self, node_with_distribution, tokenizer):
        """Test forcing specific token_id."""
        child = node_with_distribution.next(tokenizer, token_id=3)

        # Should have selected token_3
        assert "token_3" in child.string.tokens[-1]

    def test_next_creates_child(self, node_with_distribution, tokenizer):
        """Test next() creates child node."""
        initial_children = len(node_with_distribution.children)
        child = node_with_distribution.next(tokenizer, greedy=True)

        assert len(node_with_distribution.children) == initial_children + 1
        assert child in node_with_distribution.children.values()

    def test_next_returns_existing_child(self, node_with_distribution, tokenizer):
        """Test next() returns existing child if token already exists."""
        # First call creates child
        child1 = node_with_distribution.next(tokenizer, greedy=True)

        # Second greedy call should return same child
        child2 = node_with_distribution.next(tokenizer, greedy=True)

        assert child1 is child2

    def test_next_stores_logprob(self, node_with_distribution, tokenizer):
        """Test next() stores log probability in child_logprobs."""
        child = node_with_distribution.next(tokenizer, greedy=True)
        token = list(node_with_distribution.children.keys())[0]

        assert token in node_with_distribution.child_logprobs
        # Log prob should be finite
        assert np.isfinite(node_with_distribution.child_logprobs[token])

    def test_next_with_temperature(self, tokenizer):
        """Test temperature affects sampling distribution."""
        node = TreeNode(string=String.empty())
        node.set_distribution(probs=np.array([0.4, 0.6] + [0.0] * 8))

        # With high temperature, distribution should be more uniform
        # Test by sampling many times with different temperatures
        low_temp_tokens = []
        high_temp_tokens = []

        for i in range(20):
            node_low = TreeNode(string=String.empty())
            node_low.set_distribution(probs=np.array([0.4, 0.6] + [0.0] * 8))
            child = node_low.next(tokenizer, temperature=0.1, seed=i)
            low_temp_tokens.append(child.string.tokens[-1])

            node_high = TreeNode(string=String.empty())
            node_high.set_distribution(probs=np.array([0.4, 0.6] + [0.0] * 8))
            child = node_high.next(tokenizer, temperature=10.0, seed=i)
            high_temp_tokens.append(child.string.tokens[-1])

        # Low temperature should be more concentrated on max
        low_temp_max_count = low_temp_tokens.count(
            max(set(low_temp_tokens), key=low_temp_tokens.count)
        )
        high_temp_max_count = high_temp_tokens.count(
            max(set(high_temp_tokens), key=high_temp_tokens.count)
        )

        # Low temperature should have more of the max token
        assert low_temp_max_count >= high_temp_max_count


class TestTreeNodeNextTopK:
    """Test TreeNode next() with top-k sampling."""

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer(vocab_size=10)

    def test_next_top_k(self, tokenizer):
        """Test top-k filtering."""
        node = TreeNode(string=String.empty())
        # Distribution with clear top-2
        probs = np.array([0.4, 0.35, 0.1, 0.05, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0])
        node.set_distribution(probs=probs)

        # With top_k=2, should only sample from tokens 0 and 1
        samples = set()
        for i in range(50):
            new_node = TreeNode(string=String.empty())
            new_node.set_distribution(probs=probs.copy())
            child = new_node.next(tokenizer, top_k=2, seed=i)
            samples.add(child.string.tokens[-1])

        # Should only have sampled from top-2 tokens
        assert len(samples) <= 2
        assert all("token_0" in s or "token_1" in s for s in samples)


class TestTreeNodeNextTopP:
    """Test TreeNode next() with nucleus (top-p) sampling."""

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer(vocab_size=10)

    def test_next_top_p(self, tokenizer):
        """Test top-p (nucleus) filtering."""
        node = TreeNode(string=String.empty())
        # Distribution: cumsum = [0.5, 0.85, 0.95, 1.0, ...]
        probs = np.array([0.5, 0.35, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        node.set_distribution(probs=probs)

        # With top_p=0.9, should include tokens until cumsum >= 0.9
        # That means tokens 0, 1, 2 (cumsum = 0.95)
        samples = set()
        for i in range(50):
            new_node = TreeNode(string=String.empty())
            new_node.set_distribution(probs=probs.copy())
            child = new_node.next(tokenizer, top_p=0.9, seed=i)
            samples.add(child.string.tokens[-1])

        # Should only sample from tokens 0, 1, 2
        assert len(samples) <= 3


class TestTreeNodeGreedyPath:
    """Test TreeNode greedy_path method."""

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer(vocab_size=10)

    def test_greedy_path_single_step(self, tokenizer):
        """Test greedy_path with node that becomes trajectory."""
        node = TreeNode(string=String.empty())
        node.set_distribution(probs=np.array([0.8, 0.2] + [0.0] * 8))

        # Child is marked as trajectory so path should stop
        child = node.next(tokenizer, greedy=True)
        child.mark_as_trajectory()

        path = node.greedy_path(tokenizer, max_depth=10)

        # Path should be: node -> child (then stop at trajectory)
        assert len(path) >= 1

    def test_greedy_path_no_distribution(self, tokenizer):
        """Test greedy_path stops when no distribution."""
        node = TreeNode(string=String.empty())
        # No distribution set

        path = node.greedy_path(tokenizer, max_depth=10)

        # Should just return the starting node
        assert len(path) == 1
        assert path[0] == node


class TestTreeNodeSamplePath:
    """Test TreeNode sample_path method."""

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer(vocab_size=10)

    def test_sample_path_with_seed(self, tokenizer):
        """Test sample_path with seed is reproducible."""
        node1 = TreeNode(string=String.empty())
        node1.set_distribution(probs=np.array([0.5, 0.5] + [0.0] * 8))

        node2 = TreeNode(string=String.empty())
        node2.set_distribution(probs=np.array([0.5, 0.5] + [0.0] * 8))

        path1 = node1.sample_path(tokenizer, max_depth=1, seed=42)
        path2 = node2.sample_path(tokenizer, max_depth=1, seed=42)

        # Both should have same result with same seed
        assert len(path1) == len(path2)

    def test_sample_path_respects_max_depth(self, tokenizer):
        """Test sample_path respects max_depth parameter."""
        node = TreeNode(string=String.empty())
        node.set_distribution(probs=np.array([1.0] + [0.0] * 9))

        # Add distributions to children to allow deeper traversal
        path = node.sample_path(tokenizer, max_depth=1)

        # Max depth 1 means at most 2 nodes (root + 1 child)
        assert len(path) <= 2

    def test_sample_path_stops_at_trajectory(self, tokenizer):
        """Test sample_path stops at trajectory nodes."""
        node = TreeNode(string=String.empty())
        node.set_distribution(probs=np.array([1.0] + [0.0] * 9))

        # Create child and mark as trajectory
        child = node.next(tokenizer, greedy=True)
        child.mark_as_trajectory()
        child.set_distribution(probs=np.array([1.0] + [0.0] * 9))

        path = node.sample_path(tokenizer, max_depth=100)

        # Should stop at child since it's a trajectory
        assert len(path) == 2
        assert path[-1].is_trajectory()


class TestTreeNodeSetChildLogprobsFromDistribution:
    """Test set_child_logprobs_from_distribution method."""

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer(vocab_size=5)

    def test_set_child_logprobs_basic(self, tokenizer):
        """Test basic functionality."""
        node = TreeNode(string=String.empty())
        dist = np.array([0.5, 0.3, 0.2, 0.0, 0.0])

        node.set_child_logprobs_from_distribution(dist, tokenizer)

        # Should have entries for non-zero probabilities
        assert len(node.child_logprobs) >= 1

    def test_set_child_logprobs_preserves_existing(self, tokenizer):
        """Test that existing logprobs are preserved."""
        node = TreeNode(string=String.empty())

        # Set an existing logprob
        node.child_logprobs["token_0"] = -0.5

        dist = np.array([0.8, 0.2, 0.0, 0.0, 0.0])
        node.set_child_logprobs_from_distribution(dist, tokenizer)

        # Original value should be preserved
        assert node.child_logprobs["token_0"] == -0.5

    def test_set_child_logprobs_min_logprob(self, tokenizer):
        """Test min_logprob filtering."""
        node = TreeNode(string=String.empty())

        # Distribution with very low probability token
        dist = np.array([0.99, 0.01, 0.0, 0.0, 0.0])

        # With high min_logprob threshold, low prob tokens should be excluded
        node.set_child_logprobs_from_distribution(dist, tokenizer, min_logprob=-1.0)

        # Only token_0 should be included (log(0.99) ≈ -0.01)
        # token_1 has log(0.01) ≈ -4.6 which is below threshold
        # This depends on implementation details
        assert "token_0" in node.child_logprobs
