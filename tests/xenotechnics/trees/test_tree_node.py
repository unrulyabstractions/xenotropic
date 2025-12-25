"""
Tests for TreeNode class.

Tests for xenotechnics/trees/tree.py - TreeNode
"""

from __future__ import annotations

import numpy as np
import pytest

from xenotechnics.common import String
from xenotechnics.trees.tree import TreeNode


class TestTreeNodeCreation:
    """Test TreeNode creation and initialization."""

    def test_create_root_node(self):
        """Test creating root node with empty string."""
        node = TreeNode(string=String.empty())
        assert len(node.string) == 0
        assert node.parent is None
        assert node.children == {}
        assert node.child_logprobs == {}
        assert node.is_leaf()
        assert not node.is_trajectory()

    def test_create_node_with_string(self):
        """Test creating node with specific string."""
        string = String(tokens=("Hello", " ", "world"))
        node = TreeNode(string=string)
        assert node.string == string
        assert len(node.string) == 3

    def test_create_node_with_parent(self):
        """Test creating node with parent."""
        parent = TreeNode(string=String.empty())
        child = TreeNode(string=String(tokens=("a",)), parent=parent)
        assert child.parent == parent

    def test_create_node_with_metadata(self):
        """Test creating node with metadata."""
        node = TreeNode(
            string=String.empty(),
            metadata={"token_id": 42, "custom": "value"},
        )
        assert node.metadata["token_id"] == 42
        assert node.metadata["custom"] == "value"


class TestTreeNodeProperties:
    """Test TreeNode property methods."""

    def test_is_leaf_true(self):
        """Test is_leaf returns True for leaf nodes."""
        node = TreeNode(string=String.empty())
        assert node.is_leaf()

    def test_is_leaf_false(self):
        """Test is_leaf returns False for nodes with children."""
        parent = TreeNode(string=String.empty())
        parent.add_child("a", logprob=-0.5)
        assert not parent.is_leaf()

    def test_is_trajectory_default_false(self):
        """Test is_trajectory is False by default."""
        node = TreeNode(string=String.empty())
        assert not node.is_trajectory()

    def test_mark_as_trajectory(self):
        """Test mark_as_trajectory sets flag."""
        node = TreeNode(string=String.empty())
        node.mark_as_trajectory()
        assert node.is_trajectory()

    def test_depth_root(self):
        """Test depth of root node is 0."""
        root = TreeNode(string=String.empty())
        assert root.depth() == 0

    def test_depth_children(self):
        """Test depth increases with each level."""
        root = TreeNode(string=String.empty())
        child1 = root.add_child("a", logprob=-0.5)
        child2 = child1.add_child("b", logprob=-0.5)
        child3 = child2.add_child("c", logprob=-0.5)

        assert root.depth() == 0
        assert child1.depth() == 1
        assert child2.depth() == 2
        assert child3.depth() == 3

    def test_token_id_property(self):
        """Test token_id property from metadata."""
        node = TreeNode(string=String.empty(), metadata={"token_id": 42})
        assert node.token_id == 42

    def test_token_id_property_missing(self):
        """Test token_id property when not in metadata."""
        node = TreeNode(string=String.empty())
        assert node.token_id is None


class TestTreeNodeAddChild:
    """Test TreeNode add_child method."""

    def test_add_child_basic(self):
        """Test basic child addition."""
        root = TreeNode(string=String.empty())
        child = root.add_child("a", logprob=-0.5)

        assert "a" in root.children
        assert root.children["a"] == child
        assert root.child_logprobs["a"] == -0.5
        assert child.parent == root
        assert child.string.tokens == ("a",)

    def test_add_child_with_token_id(self):
        """Test adding child with token ID."""
        root = TreeNode(string=String.empty())
        child = root.add_child("a", logprob=-0.5, token_id=42)

        assert child.metadata["token_id"] == 42

    def test_add_child_with_metadata(self):
        """Test adding child with custom metadata."""
        root = TreeNode(string=String.empty())
        child = root.add_child("a", logprob=-0.5, metadata={"custom": "value"})

        assert child.metadata["custom"] == "value"

    def test_add_child_updates_existing(self):
        """Test adding child with same token updates logprob."""
        root = TreeNode(string=String.empty())
        child1 = root.add_child("a", logprob=-0.5)
        child2 = root.add_child("a", logprob=-1.0)  # Same token

        assert child1 == child2  # Same object
        assert root.child_logprobs["a"] == -1.0  # Updated logprob

    def test_add_multiple_children(self):
        """Test adding multiple children."""
        root = TreeNode(string=String.empty())
        child_a = root.add_child("a", logprob=-0.5)
        child_b = root.add_child("b", logprob=-1.0)
        child_c = root.add_child("c", logprob=-1.5)

        assert len(root.children) == 3
        assert root.children["a"] == child_a
        assert root.children["b"] == child_b
        assert root.children["c"] == child_c


class TestTreeNodePathMethods:
    """Test TreeNode path-related methods."""

    @pytest.fixture
    def simple_tree(self):
        """Create simple tree for testing."""
        root = TreeNode(string=String.empty())
        child_a = root.add_child("a", logprob=-0.5, token_id=1)
        child_b = root.add_child("b", logprob=-1.0, token_id=2)
        child_ax = child_a.add_child("x", logprob=-0.3, token_id=10)
        child_ay = child_a.add_child("y", logprob=-0.7, token_id=11)
        child_bz = child_b.add_child("z", logprob=-0.2, token_id=12)
        return root, child_a, child_b, child_ax, child_ay, child_bz

    def test_path_to_root(self, simple_tree):
        """Test path_to_root returns correct path."""
        root, child_a, child_b, child_ax, child_ay, child_bz = simple_tree

        path = child_ax.path_to_root()
        assert path == [root, child_a, child_ax]

    def test_path_to_root_from_root(self, simple_tree):
        """Test path_to_root from root returns single element."""
        root, *_ = simple_tree
        path = root.path_to_root()
        assert path == [root]

    def test_path_logprob(self, simple_tree):
        """Test path_logprob computes sum of log probabilities."""
        root, child_a, child_b, child_ax, child_ay, child_bz = simple_tree

        # Path to child_ax: root -> a (-0.5) -> x (-0.3)
        assert child_ax.path_logprob() == pytest.approx(-0.8)

        # Path to child_bz: root -> b (-1.0) -> z (-0.2)
        assert child_bz.path_logprob() == pytest.approx(-1.2)

    def test_path_logprob_root(self, simple_tree):
        """Test path_logprob of root is 0."""
        root, *_ = simple_tree
        assert root.path_logprob() == 0.0

    def test_probability(self, simple_tree):
        """Test probability is exp of path_logprob."""
        root, child_a, *_ = simple_tree

        prob = child_a.probability()
        expected = np.exp(-0.5)
        assert prob == pytest.approx(expected)


class TestTreeNodeDistribution:
    """Test TreeNode distribution methods."""

    def test_set_distribution_with_logits(self):
        """Test set_distribution with logits."""
        node = TreeNode(string=String.empty())
        logits = np.array([1.0, 2.0, 3.0])
        node.set_distribution(logits=logits)

        assert node.next_token_logits is not None
        assert node.next_token_distribution is not None
        assert len(node.next_token_distribution) == 3
        assert np.sum(node.next_token_distribution) == pytest.approx(1.0)

    def test_set_distribution_with_probs(self):
        """Test set_distribution with probabilities."""
        node = TreeNode(string=String.empty())
        probs = np.array([0.2, 0.3, 0.5])
        node.set_distribution(probs=probs)

        assert node.next_token_distribution is not None
        np.testing.assert_array_almost_equal(node.next_token_distribution, probs)

    def test_set_distribution_with_temperature(self):
        """Test set_distribution with temperature."""
        node = TreeNode(string=String.empty())
        logits = np.array([1.0, 2.0, 3.0])

        # Low temperature makes distribution more peaked
        node.set_distribution(logits=logits, temperature=0.1)
        low_temp_dist = node.next_token_distribution.copy()

        # High temperature makes distribution more uniform
        node.set_distribution(logits=logits, temperature=10.0)
        high_temp_dist = node.next_token_distribution

        # Max probability should be higher with low temperature
        assert np.max(low_temp_dist) > np.max(high_temp_dist)

    def test_set_distribution_neither_raises(self):
        """Test set_distribution raises when neither logits nor probs provided."""
        node = TreeNode(string=String.empty())
        with pytest.raises(ValueError, match="Must provide either logits or probs"):
            node.set_distribution()


class TestTreeNodeTraversal:
    """Test TreeNode traversal methods."""

    @pytest.fixture
    def tree_with_trajectories(self):
        """Create tree with trajectory nodes."""
        root = TreeNode(string=String.empty())

        # Path 1: a -> x (trajectory)
        child_a = root.add_child("a", logprob=-0.5)
        traj_1 = child_a.add_child("x", logprob=-0.3)
        traj_1.mark_as_trajectory()

        # Path 2: a -> y (trajectory)
        traj_2 = child_a.add_child("y", logprob=-0.7)
        traj_2.mark_as_trajectory()

        # Path 3: b -> z (trajectory)
        child_b = root.add_child("b", logprob=-1.0)
        traj_3 = child_b.add_child("z", logprob=-0.2)
        traj_3.mark_as_trajectory()

        return root

    def test_get_all_descendants(self, tree_with_trajectories):
        """Test get_all_descendants returns all nodes."""
        root = tree_with_trajectories
        descendants = root.get_all_descendants()

        # 2 first-level + 3 second-level = 5 descendants
        assert len(descendants) == 5

    def test_get_trajectories(self, tree_with_trajectories):
        """Test get_trajectories returns trajectory strings."""
        root = tree_with_trajectories
        trajectories = root.get_trajectories()

        assert len(trajectories) == 3
        # Check each trajectory is a String
        for traj in trajectories:
            assert isinstance(traj, String)

    def test_get_trajectory_nodes(self, tree_with_trajectories):
        """Test get_trajectory_nodes returns trajectory nodes."""
        root = tree_with_trajectories
        traj_nodes = root.get_trajectory_nodes()

        assert len(traj_nodes) == 3
        for node in traj_nodes:
            assert node.is_trajectory()

    def test_get_child(self):
        """Test get_child returns correct child."""
        root = TreeNode(string=String.empty())
        child_a = root.add_child("a", logprob=-0.5)

        assert root.get_child("a") == child_a
        assert root.get_child("nonexistent") is None


class TestTreeNodeFindMethods:
    """Test TreeNode find methods."""

    @pytest.fixture
    def search_tree(self):
        """Create tree for search testing."""
        root = TreeNode(string=String.empty())
        child_a = root.add_child("a", logprob=-0.5)
        child_b = root.add_child("b", logprob=-1.0)
        child_ax = child_a.add_child("x", logprob=-0.3)
        child_ax.mark_as_trajectory()
        return root, child_a, child_b, child_ax

    def test_find_node(self, search_tree):
        """Test find_node finds matching prefix."""
        root, child_a, child_b, child_ax = search_tree

        # Find child_a by its string
        found = root.find_node(String(tokens=("a",)))
        assert found == child_a

        # Find child_ax
        found = root.find_node(String(tokens=("a", "x")))
        assert found == child_ax

    def test_find_node_not_found(self, search_tree):
        """Test find_node returns None when not found."""
        root, *_ = search_tree
        found = root.find_node(String(tokens=("nonexistent",)))
        assert found is None

    def test_find_trajectory_node(self, search_tree):
        """Test find_trajectory_node finds trajectory."""
        root, child_a, child_b, child_ax = search_tree

        found = root.find_trajectory_node(String(tokens=("a", "x")))
        assert found == child_ax
        assert found.is_trajectory()

    def test_find_trajectory_node_not_trajectory(self, search_tree):
        """Test find_trajectory_node returns None for non-trajectory."""
        root, child_a, *_ = search_tree

        # child_a exists but is not a trajectory
        found = root.find_trajectory_node(String(tokens=("a",)))
        assert found is None


class TestTreeNodeBranchMass:
    """Test TreeNode branch mass methods."""

    @pytest.fixture
    def mass_tree(self):
        """Create tree for branch mass testing."""
        root = TreeNode(string=String.empty())
        # Two branches with known probabilities
        child_a = root.add_child("a", logprob=np.log(0.6))
        child_b = root.add_child("b", logprob=np.log(0.4))

        # child_a has two children
        child_a.add_child("x", logprob=np.log(0.5))
        child_a.add_child("y", logprob=np.log(0.5))

        # child_b is a leaf
        return root, child_a, child_b

    def test_branch_mass_leaf(self):
        """Test branch_mass for leaf node."""
        node = TreeNode(string=String.empty())
        # Leaf with no parent has prob 1.0
        assert node.branch_mass() == pytest.approx(1.0)

    def test_branch_mass_logprob(self, mass_tree):
        """Test branch_mass_logprob computation."""
        root, child_a, child_b = mass_tree

        # child_b is a leaf, mass = 0.4
        assert child_b.branch_mass() == pytest.approx(0.4)

        # child_a has two children at depth 2
        # Each has prob 0.6 * 0.5 = 0.3
        # Total mass = 0.3 + 0.3 = 0.6
        assert child_a.branch_mass() == pytest.approx(0.6)


class TestTreeNodeConditionalProbability:
    """Test TreeNode conditional probability methods."""

    @pytest.fixture
    def conditional_tree(self):
        """Create tree for conditional probability testing."""
        # Tree: prompt (2 tokens) + continuation (2 tokens)
        root = TreeNode(string=String.empty())
        t1 = root.add_child("The", logprob=-0.1)
        t2 = t1.add_child(" ", logprob=-0.2)
        # After prompt
        t3 = t2.add_child("cat", logprob=-0.3)
        t4 = t3.add_child(".", logprob=-0.4)
        t4.mark_as_trajectory()
        return root, t4

    def test_get_continuation_logprob(self, conditional_tree):
        """Test get_continuation_logprob after prompt."""
        root, traj = conditional_tree

        # Prompt has 2 tokens, continuation has 2 tokens
        # Continuation logprob = -0.3 + -0.4 = -0.7
        logprob = traj.get_continuation_logprob(prompt_token_count=2)
        assert logprob == pytest.approx(-0.7)

    def test_get_continuation_prob(self, conditional_tree):
        """Test get_continuation_prob."""
        root, traj = conditional_tree

        prob = traj.get_continuation_prob(prompt_token_count=2)
        expected = np.exp(-0.7)
        assert prob == pytest.approx(expected)

    def test_get_conditional_logprob(self, conditional_tree):
        """Test get_conditional_logprob with prompt String."""
        root, traj = conditional_tree
        prompt = String(tokens=("The", " "))

        logprob = traj.get_conditional_logprob(prompt)
        assert logprob == pytest.approx(-0.7)

    def test_get_conditional_probabilities(self, conditional_tree):
        """Test get_conditional_probabilities for multiple trajectories."""
        root, traj = conditional_tree

        # Add another trajectory
        t1 = root.get_child("The")
        t2 = t1.get_child(" ")
        t3_alt = t2.add_child("dog", logprob=-0.5)
        t4_alt = t3_alt.add_child("!", logprob=-0.6)
        t4_alt.mark_as_trajectory()

        prompt = String(tokens=("The", " "))
        trajectory_nodes = root.get_trajectory_nodes()

        probs = root.get_conditional_probabilities(trajectory_nodes, prompt)
        assert len(probs) == 2
        assert np.all(probs > 0)


class TestTreeNodeRepr:
    """Test TreeNode string representations."""

    def test_repr(self):
        """Test __repr__ format."""
        node = TreeNode(string=String(tokens=("test",)))
        rep = repr(node)

        assert "TreeNode" in rep
        assert "string=" in rep
        assert "depth=" in rep
        assert "prob=" in rep

    def test_str(self):
        """Test __str__ format."""
        node = TreeNode(string=String(tokens=("test",)))
        s = str(node)

        assert "TreeNode:" in s
        assert "Text:" in s
        assert "Tokens:" in s
        assert "Depth:" in s

    def test_str_with_distribution(self):
        """Test __str__ with distribution."""
        node = TreeNode(string=String(tokens=("test",)))
        node.set_distribution(probs=np.array([0.2, 0.3, 0.5]))
        s = str(node)

        assert "entropy=" in s
        assert "top5:" in s
