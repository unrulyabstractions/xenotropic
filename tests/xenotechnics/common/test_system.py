"""
Tests for abstract system interface.

Tests for xenotechnics/common/system.py
"""

from __future__ import annotations

import numpy as np
import pytest

from xenotechnics.common import FunctionalStructure, String
from xenotechnics.systems.vector_system import VectorSystem
from xenotechnics.trees.tree import LLMTree


class TestAbstractSystemCore:
    """Tests for AbstractSystem.core() method."""

    @pytest.fixture
    def simple_system(self):
        """Create a simple vector system for testing."""
        structures = [
            FunctionalStructure(lambda s: min(1.0, len(s.tokens) / 5), name="length"),
            FunctionalStructure(lambda s: 0.5, name="constant"),
        ]
        return VectorSystem(structures)

    @pytest.fixture
    def tree_with_trajectories(self):
        """Create tree with trajectories."""
        LLMTree.clear_all_trees()
        tree = LLMTree.get_tree("test_core")

        # Build tree: root -> a -> x (trajectory), y (trajectory)
        child_a = tree.root.add_child("a", logprob=np.log(0.6), token_id=1)
        child_ax = child_a.add_child("x", logprob=np.log(0.7), token_id=10)
        child_ax.mark_as_trajectory()

        child_ay = child_a.add_child("y", logprob=np.log(0.3), token_id=11)
        child_ay.mark_as_trajectory()

        return tree

    @pytest.fixture
    def empty_tree(self):
        """Create tree with no trajectories."""
        LLMTree.clear_all_trees()
        tree = LLMTree.get_tree("test_empty")
        # Add nodes but don't mark as trajectory
        tree.root.add_child("a", logprob=-0.1, token_id=1)
        return tree

    def test_core_with_no_trajectories_raises(self, simple_system, empty_tree):
        """Test that core() raises ValueError when no trajectories found."""
        prompt = String.empty()

        with pytest.raises(ValueError, match="No trajectories found in tree"):
            simple_system.core(empty_tree.root, prompt)

    def test_core_with_all_trajectories(self, simple_system, tree_with_trajectories):
        """Test core() using all trajectories in tree."""
        tree = tree_with_trajectories
        prompt = String.empty()

        core = simple_system.core(tree.root, prompt)

        # Should return valid compliance
        assert core is not None
        assert core.string is None  # Core has no string
        assert len(core) == 2

    def test_core_with_specific_trajectories(
        self, simple_system, tree_with_trajectories
    ):
        """Test core() with specific trajectories provided."""
        tree = tree_with_trajectories
        prompt = String.empty()

        # Get one specific trajectory
        traj_nodes = tree.root.get_trajectory_nodes()
        specific_trajectories = [traj_nodes[0].string]

        core = simple_system.core(tree.root, prompt, trajectories=specific_trajectories)

        # Should return valid compliance
        assert core is not None
        assert core.string is None
        assert len(core) == 2

    def test_core_with_trajectory_not_found_raises(
        self, simple_system, tree_with_trajectories
    ):
        """Test that core() raises ValueError when trajectory not in tree."""
        tree = tree_with_trajectories
        prompt = String.empty()

        # Create a trajectory that doesn't exist in tree
        fake_trajectory = String(tokens=("not", "in", "tree"))

        with pytest.raises(ValueError, match="Trajectory not found in tree"):
            simple_system.core(tree.root, prompt, trajectories=[fake_trajectory])

    def test_core_with_prompt(self, simple_system, tree_with_trajectories):
        """Test core() with non-empty prompt."""
        tree = tree_with_trajectories
        prompt = String(tokens=("a",))

        core = simple_system.core(tree.root, prompt)

        # Should compute conditional probabilities correctly
        assert core is not None

    def test_core_with_multiple_specific_trajectories(self, simple_system):
        """Test core() with multiple specific trajectories."""
        LLMTree.clear_all_trees()
        tree = LLMTree.get_tree("test_multi_traj")

        # Build tree with 3 trajectories
        child_a = tree.root.add_child("a", logprob=np.log(0.5), token_id=1)
        child_a.mark_as_trajectory()

        child_b = tree.root.add_child("b", logprob=np.log(0.3), token_id=2)
        child_b.mark_as_trajectory()

        child_c = tree.root.add_child("c", logprob=np.log(0.2), token_id=3)
        child_c.mark_as_trajectory()

        prompt = String.empty()

        # Get all trajectories
        all_trajs = tree.root.get_trajectory_nodes()

        # Use only 2 of the 3 trajectories
        specific = [all_trajs[0].string, all_trajs[1].string]

        core = simple_system.core(tree.root, prompt, trajectories=specific)

        assert core is not None
        assert len(core) == 2


class TestAbstractSystemInterface:
    """Tests for AbstractSystem interface methods."""

    @pytest.fixture
    def system(self):
        """Create a simple system for testing."""
        structures = [
            FunctionalStructure(lambda s: 0.5, name="s1"),
            FunctionalStructure(lambda s: 0.7, name="s2"),
        ]
        return VectorSystem(structures)

    def test_len_returns_structure_count(self, system):
        """Test __len__ returns number of structures."""
        assert len(system) == 2

    def test_structure_names_returns_list(self, system):
        """Test structure_names returns list of names."""
        names = system.structure_names()
        assert names == ["s1", "s2"]

    def test_score_operator_returns_operator(self, system):
        """Test score_operator property."""
        op = system.score_operator
        assert op is not None

    def test_difference_operator_returns_operator(self, system):
        """Test difference_operator property."""
        op = system.difference_operator
        assert op is not None

    def test_compliance_returns_compliance_object(self, system):
        """Test compliance returns proper object."""
        string = String(tokens=("test",))
        compliance = system.compliance(string)

        assert compliance is not None
        assert compliance.string == string
        assert len(compliance) == 2

    def test_compute_core_returns_compliance(self, system):
        """Test compute_core returns compliance object."""
        trajectories = [
            String(tokens=("a", "b")),
            String(tokens=("c", "d")),
        ]
        probs = np.array([0.6, 0.4])

        core = system.compute_core(trajectories, probs)

        assert core is not None
        assert core.string is None
        assert len(core) == 2
