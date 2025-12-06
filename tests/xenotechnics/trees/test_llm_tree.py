"""
Tests for LLMTree class.

Tests for xenotechnics/trees/tree.py - LLMTree
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from xenotechnics.common import String
from xenotechnics.trees.tree import LLMTree, TreeNode


class TestLLMTreeCreation:
    """Test LLMTree creation and singleton pattern."""

    def setup_method(self):
        """Clear all trees before each test."""
        LLMTree.clear_all_trees()

    def test_create_tree(self):
        """Test creating new LLMTree."""
        tree = LLMTree("test_llm")
        assert tree.llm_id == "test_llm"
        assert tree.root is not None
        assert isinstance(tree.root, TreeNode)
        assert tree.root.string == String.empty()

    def test_create_tree_with_tokenizer(self):
        """Test creating tree with tokenizer."""
        tokenizer = MagicMock()
        tree = LLMTree("test_llm", tokenizer=tokenizer)
        assert tree.tokenizer == tokenizer

    def test_create_tree_with_system(self):
        """Test creating tree with system."""
        system = MagicMock()
        tree = LLMTree("test_llm", system=system)
        assert tree.system == system


class TestLLMTreeSingleton:
    """Test LLMTree singleton pattern."""

    def setup_method(self):
        """Clear all trees before each test."""
        LLMTree.clear_all_trees()

    def test_get_tree_creates_new(self):
        """Test get_tree creates new tree if not exists."""
        tree = LLMTree.get_tree("new_llm")
        assert tree is not None
        assert tree.llm_id == "new_llm"

    def test_get_tree_returns_existing(self):
        """Test get_tree returns existing tree."""
        tree1 = LLMTree.get_tree("same_llm")
        tree2 = LLMTree.get_tree("same_llm")
        assert tree1 is tree2

    def test_get_tree_reset(self):
        """Test get_tree with reset=True creates new tree."""
        tree1 = LLMTree.get_tree("reset_test")
        # Add something to the tree
        tree1.root.add_child("a", logprob=-0.5)

        tree2 = LLMTree.get_tree("reset_test", reset=True)
        assert tree2 is not tree1
        assert len(tree2.root.children) == 0  # Fresh tree

    def test_get_tree_updates_tokenizer(self):
        """Test get_tree updates tokenizer on existing tree."""
        tree1 = LLMTree.get_tree("update_test")
        assert tree1.tokenizer is None

        tokenizer = MagicMock()
        tree2 = LLMTree.get_tree("update_test", tokenizer=tokenizer)
        assert tree2.tokenizer == tokenizer
        assert tree1 is tree2  # Same object

    def test_get_tree_updates_system(self):
        """Test get_tree updates system on existing tree."""
        tree1 = LLMTree.get_tree("update_test")
        assert tree1.system is None

        system = MagicMock()
        tree2 = LLMTree.get_tree("update_test", system=system)
        assert tree2.system == system

    def test_different_llm_ids_different_trees(self):
        """Test different LLM IDs get different trees."""
        tree1 = LLMTree.get_tree("llm_a")
        tree2 = LLMTree.get_tree("llm_b")

        assert tree1 is not tree2
        assert tree1.llm_id != tree2.llm_id


class TestLLMTreeClear:
    """Test LLMTree clear methods."""

    def setup_method(self):
        """Clear all trees before each test."""
        LLMTree.clear_all_trees()

    def test_clear_tree(self):
        """Test clear_tree removes specific tree."""
        LLMTree.get_tree("to_clear")
        assert LLMTree.clear_tree("to_clear")
        assert "to_clear" not in LLMTree.list_llms()

    def test_clear_tree_nonexistent(self):
        """Test clear_tree returns False for nonexistent."""
        assert not LLMTree.clear_tree("nonexistent")

    def test_clear_all_trees(self):
        """Test clear_all_trees removes all trees."""
        LLMTree.get_tree("llm1")
        LLMTree.get_tree("llm2")
        LLMTree.get_tree("llm3")

        LLMTree.clear_all_trees()
        assert LLMTree.list_llms() == []

    def test_list_llms(self):
        """Test list_llms returns all LLM IDs."""
        LLMTree.get_tree("alpha")
        LLMTree.get_tree("beta")
        LLMTree.get_tree("gamma")

        llms = LLMTree.list_llms()
        assert set(llms) == {"alpha", "beta", "gamma"}


class TestLLMTreeMethods:
    """Test LLMTree instance methods."""

    def setup_method(self):
        """Clear all trees before each test."""
        LLMTree.clear_all_trees()

    @pytest.fixture
    def populated_tree(self):
        """Create tree with nodes for testing."""
        tree = LLMTree.get_tree("test")

        # Add some nodes
        child_a = tree.root.add_child("a", logprob=np.log(0.6))
        child_b = tree.root.add_child("b", logprob=np.log(0.4))

        child_ax = child_a.add_child("x", logprob=np.log(0.7))
        child_ax.mark_as_trajectory()

        child_ay = child_a.add_child("y", logprob=np.log(0.3))
        child_ay.mark_as_trajectory()

        child_bz = child_b.add_child("z", logprob=np.log(1.0))
        child_bz.mark_as_trajectory()

        return tree

    def test_get_trajectories(self, populated_tree):
        """Test get_trajectories returns all trajectory strings."""
        trajectories = populated_tree.get_trajectories()
        assert len(trajectories) == 3

        # Check trajectory contents
        trajectory_texts = {t.to_text() for t in trajectories}
        assert "ax" in trajectory_texts
        assert "ay" in trajectory_texts
        assert "bz" in trajectory_texts

    def test_total_mass(self, populated_tree):
        """Test total_mass computes correctly."""
        mass = populated_tree.total_mass()
        # All probability mass is in the three trajectories
        # mass = 0.6*0.7 + 0.6*0.3 + 0.4*1.0 = 0.42 + 0.18 + 0.4 = 1.0
        assert mass == pytest.approx(1.0)

    def test_max_depth(self, populated_tree):
        """Test max_depth returns correct value."""
        assert populated_tree.max_depth() == 2

    def test_max_depth_empty_tree(self):
        """Test max_depth on empty tree is 0."""
        tree = LLMTree.get_tree("empty")
        assert tree.max_depth() == 0

    def test_get_node(self, populated_tree):
        """Test get_node finds node by prefix."""
        node = populated_tree.get_node(String(tokens=("a",)))
        assert node is not None
        assert node.string.to_text() == "a"

    def test_get_node_not_found(self, populated_tree):
        """Test get_node returns None when not found."""
        node = populated_tree.get_node(String(tokens=("nonexistent",)))
        assert node is None


class TestLLMTreeRepr:
    """Test LLMTree string representation."""

    def setup_method(self):
        """Clear all trees before each test."""
        LLMTree.clear_all_trees()

    def test_repr(self):
        """Test __repr__ format."""
        tree = LLMTree.get_tree("test_repr")
        rep = repr(tree)

        assert "LLMTree" in rep
        assert "test_repr" in rep
        assert "total_mass=" in rep
        assert "max_depth=" in rep


class TestLLMTreeIntegration:
    """Integration tests for LLMTree with TreeNode operations."""

    def setup_method(self):
        """Clear all trees before each test."""
        LLMTree.clear_all_trees()

    def test_build_tree_from_generations(self):
        """Test building tree from simulated generations."""
        tree = LLMTree.get_tree("generation_test")

        # Simulate multiple generation runs
        for _ in range(3):
            # Start from root
            node = tree.root

            # Add tokens
            node = node.add_child("The", logprob=-0.1)
            node = node.add_child(" ", logprob=-0.2)
            node = node.add_child("cat", logprob=-0.3)
            node.mark_as_trajectory()

        # Should have only one trajectory (same path)
        assert len(tree.get_trajectories()) == 1

    def test_multiple_trajectories_same_prefix(self):
        """Test multiple trajectories with same prefix."""
        tree = LLMTree.get_tree("prefix_test")

        # Shared prefix
        prefix_node = tree.root.add_child("The", logprob=-0.1)
        prefix_node = prefix_node.add_child(" ", logprob=-0.2)

        # Different continuations
        cat = prefix_node.add_child("cat", logprob=-0.3)
        cat.mark_as_trajectory()

        dog = prefix_node.add_child("dog", logprob=-0.4)
        dog.mark_as_trajectory()

        assert len(tree.get_trajectories()) == 2

    def test_tree_preserves_structure(self):
        """Test tree structure is preserved across operations."""
        tree = LLMTree.get_tree("structure_test")

        # Build tree
        a = tree.root.add_child("a", logprob=-0.5)
        b = a.add_child("b", logprob=-0.3)
        c = b.add_child("c", logprob=-0.2)
        c.mark_as_trajectory()

        # Verify structure
        found = tree.get_node(String(tokens=("a", "b", "c")))
        assert found == c
        assert found.is_trajectory()
        assert found.path_logprob() == pytest.approx(-1.0)
