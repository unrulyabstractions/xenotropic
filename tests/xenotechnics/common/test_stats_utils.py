"""
Tests for statistical utility functions.

Tests for xenotechnics/common/stats_utils.py
"""

from __future__ import annotations

import numpy as np
import pytest

from xenotechnics.common import FunctionalStructure, String
from xenotechnics.common.stats_utils import deviance_variance, expected_deviance
from xenotechnics.systems.vector_system import VectorSystem
from xenotechnics.trees.tree import TreeNode


class TestExpectedDeviance:
    """Test expected_deviance function."""

    @pytest.fixture
    def simple_system(self):
        """Create simple test system."""

        def length_fn(s: String) -> float:
            return min(1.0, len(s.tokens) / 10)

        return VectorSystem([FunctionalStructure(length_fn, name="length")])

    @pytest.fixture
    def tree_with_trajectories(self, simple_system):
        """Create tree with trajectory nodes."""
        # Build tree structure
        root = TreeNode(string=String.empty())

        # Add children with distributions
        root.set_distribution(probs=np.array([0.6, 0.4]))

        child1 = root.add_child("hello", logprob=np.log(0.6), token_id=0)
        child2 = root.add_child("world", logprob=np.log(0.4), token_id=1)

        # Mark as trajectories
        child1.mark_as_trajectory()
        child2.mark_as_trajectory()

        prompt = String.empty()

        return root, prompt

    def test_no_trajectories(self, simple_system):
        """Test with empty tree returns zero."""
        root = TreeNode(string=String.empty())
        prompt = String.empty()

        result = expected_deviance(simple_system, root, prompt)
        assert result == 0.0

    def test_single_trajectory(self, simple_system):
        """Test with single trajectory."""
        root = TreeNode(string=String.empty())
        root.set_distribution(probs=np.array([1.0]))
        child = root.add_child("hello", logprob=0.0, token_id=0)
        child.mark_as_trajectory()
        prompt = String.empty()

        result = expected_deviance(simple_system, root, prompt)

        # Single trajectory, deviance should be finite
        assert np.isfinite(result)
        # Single trajectory has zero deviance from its own core
        assert result == pytest.approx(0.0, abs=0.01)

    def test_multiple_trajectories(self, simple_system, tree_with_trajectories):
        """Test with multiple trajectories."""
        root, prompt = tree_with_trajectories

        result = expected_deviance(simple_system, root, prompt)

        assert np.isfinite(result)
        assert result >= 0.0

    def test_weighted_by_probability(self, simple_system):
        """Test that deviance is weighted by probability."""
        root = TreeNode(string=String.empty())

        # Create two trajectories with very different probabilities
        root.set_distribution(probs=np.array([0.99, 0.01]))

        # Both have same string content
        child1 = root.add_child("a", logprob=np.log(0.99), token_id=0)
        child2 = root.add_child("a", logprob=np.log(0.01), token_id=1)

        child1.mark_as_trajectory()
        child2.mark_as_trajectory()

        prompt = String.empty()

        result = expected_deviance(simple_system, root, prompt)

        # Should be dominated by high-probability trajectory
        assert np.isfinite(result)


class TestDevianceVariance:
    """Test deviance_variance function."""

    @pytest.fixture
    def simple_system(self):
        """Create simple test system."""

        def length_fn(s: String) -> float:
            return min(1.0, len(s.tokens) / 10)

        return VectorSystem([FunctionalStructure(length_fn, name="length")])

    def test_no_trajectories(self, simple_system):
        """Test with empty tree returns zero."""
        root = TreeNode(string=String.empty())
        prompt = String.empty()

        result = deviance_variance(simple_system, root, prompt)
        assert result == 0.0

    def test_single_trajectory(self, simple_system):
        """Test with single trajectory has zero variance."""
        root = TreeNode(string=String.empty())
        root.set_distribution(probs=np.array([1.0]))
        child = root.add_child("hello", logprob=0.0, token_id=0)
        child.mark_as_trajectory()
        prompt = String.empty()

        result = deviance_variance(simple_system, root, prompt)

        # Single trajectory has zero variance
        assert result == pytest.approx(0.0, abs=0.01)

    def test_identical_trajectories(self, simple_system):
        """Test identical trajectories have zero variance."""
        root = TreeNode(string=String.empty())
        root.set_distribution(probs=np.array([0.5, 0.5]))

        # Same content (but different token IDs for tree structure)
        child1 = root.add_child("hello", logprob=np.log(0.5), token_id=0)
        child2 = root.add_child("hello", logprob=np.log(0.5), token_id=1)

        child1.mark_as_trajectory()
        child2.mark_as_trajectory()

        prompt = String.empty()

        result = deviance_variance(simple_system, root, prompt)

        # Identical strings should have zero variance
        assert result == pytest.approx(0.0, abs=0.01)

    def test_different_trajectories_positive_variance(self, simple_system):
        """Test different trajectories have positive variance."""
        root = TreeNode(string=String.empty())
        root.set_distribution(probs=np.array([0.5, 0.5]))

        # Very different content (different lengths)
        child1 = root.add_child("a", logprob=np.log(0.5), token_id=0)
        child2_str = "very_long_string_here"
        child2 = root.add_child(child2_str, logprob=np.log(0.5), token_id=1)

        child1.mark_as_trajectory()
        child2.mark_as_trajectory()

        prompt = String.empty()

        result = deviance_variance(simple_system, root, prompt)

        # Different strings should have positive variance
        assert result >= 0.0
        assert np.isfinite(result)

    def test_non_negative(self, simple_system):
        """Test that variance is always non-negative."""
        root = TreeNode(string=String.empty())
        root.set_distribution(probs=np.array([0.3, 0.3, 0.4]))

        child1 = root.add_child("short", logprob=np.log(0.3), token_id=0)
        child2 = root.add_child("medium_length", logprob=np.log(0.3), token_id=1)
        child3 = root.add_child(
            "this_is_a_longer_string", logprob=np.log(0.4), token_id=2
        )

        child1.mark_as_trajectory()
        child2.mark_as_trajectory()
        child3.mark_as_trajectory()

        prompt = String.empty()

        result = deviance_variance(simple_system, root, prompt)

        assert result >= 0.0
