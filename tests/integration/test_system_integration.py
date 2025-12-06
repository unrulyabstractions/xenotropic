"""
Integration tests for xenotropic system components.

Tests that verify different modules work together correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from xenotechnics.common import (
    CompositeStructure,
    FunctionalStructure,
    String,
)
from xenotechnics.operators import (
    L1DifferenceOperator,
    L1ScoreOperator,
    L2DifferenceOperator,
    L2ScoreOperator,
    MeanScoreOperator,
)
from xenotechnics.systems.vector_system import (
    VectorOrientation,
    VectorSystem,
    VectorSystemCompliance,
    compute_core_from_trajectories,
)
from xenotechnics.trees.tree import LLMTree


class TestSystemPipeline:
    """Test full system evaluation pipeline."""

    @pytest.fixture
    def text_analysis_system(self):
        """Create system for text analysis."""

        def length_score(s: String) -> float:
            """Score based on length (longer = higher)."""
            return min(1.0, len(s.tokens) / 20)

        def diversity_score(s: String) -> float:
            """Score based on unique tokens."""
            if len(s.tokens) == 0:
                return 0.0
            unique = len(set(s.tokens))
            return unique / len(s.tokens)

        def vowel_ratio(s: String) -> float:
            """Ratio of vowels in text."""
            text = s.to_text().lower()
            if not text:
                return 0.5
            vowels = sum(1 for c in text if c in "aeiou")
            return vowels / len(text)

        structures = [
            FunctionalStructure(length_score, name="length"),
            FunctionalStructure(diversity_score, name="diversity"),
            FunctionalStructure(vowel_ratio, name="vowels"),
        ]

        return VectorSystem(
            structures,
            score_operator=MeanScoreOperator(),
            difference_operator=L2DifferenceOperator(),
        )

    def test_single_string_evaluation(self, text_analysis_system):
        """Test evaluating a single string."""
        system = text_analysis_system
        string = String(tokens=("The", " ", "quick", " ", "brown", " ", "fox"))

        compliance = system.compliance(string)

        assert isinstance(compliance, VectorSystemCompliance)
        assert len(compliance) == 3

        # All scores should be in [0, 1]
        values = compliance.to_array()
        assert np.all(values >= 0)
        assert np.all(values <= 1)

    def test_core_computation_pipeline(self, text_analysis_system):
        """Test computing core from trajectories."""
        system = text_analysis_system

        trajectories = [
            String(tokens=("Hello", " ", "world")),
            String(tokens=("The", " ", "cat", " ", "sat")),
            String(tokens=("Quick", " ", "test")),
        ]
        probs = np.array([0.5, 0.3, 0.2])

        core = compute_core_from_trajectories(system, trajectories, probs)

        assert isinstance(core, VectorSystemCompliance)
        assert core.string is None  # Core has no string

        # Core values should be in [0, 1]
        values = core.to_array()
        assert np.all(values >= 0)
        assert np.all(values <= 1)

    def test_orientation_computation(self, text_analysis_system):
        """Test computing orientation between string and core."""
        system = text_analysis_system

        # Compute core
        trajectories = [
            String(tokens=("a", "b", "c")),
            String(tokens=("x", "y", "z")),
        ]
        probs = np.array([0.5, 0.5])
        core = compute_core_from_trajectories(system, trajectories, probs)

        # Compute compliance for specific string
        string = String(tokens=("hello", " ", "world", " ", "test"))
        compliance = system.compliance(string)

        # Compute orientation
        orientation = VectorOrientation(compliance, core, system.difference_operator)

        assert len(orientation) == 3
        deviance = orientation.deviance()
        assert deviance >= 0


class TestTreeSystemIntegration:
    """Test tree and system integration."""

    def setup_method(self):
        """Clear trees before each test."""
        LLMTree.clear_all_trees()

    def test_tree_trajectories_to_core(self):
        """Test using tree trajectories to compute core."""
        # Build a simple tree
        tree = LLMTree.get_tree("test_integration")

        # Add trajectories
        child_a = tree.root.add_child("a", logprob=np.log(0.6), token_id=1)
        child_ax = child_a.add_child("x", logprob=np.log(0.7), token_id=10)
        child_ax.mark_as_trajectory()

        child_ay = child_a.add_child("y", logprob=np.log(0.3), token_id=11)
        child_ay.mark_as_trajectory()

        child_b = tree.root.add_child("b", logprob=np.log(0.4), token_id=2)
        child_bz = child_b.add_child("z", logprob=np.log(1.0), token_id=12)
        child_bz.mark_as_trajectory()

        # Get trajectories
        trajectories = tree.get_trajectories()
        assert len(trajectories) == 3

        # Verify trajectory probabilities sum to approximately 1
        traj_nodes = tree.root.get_trajectory_nodes()
        total_prob = sum(node.probability() for node in traj_nodes)
        assert total_prob == pytest.approx(1.0, rel=0.01)

    def test_conditional_probabilities(self):
        """Test conditional probability computation."""
        tree = LLMTree.get_tree("cond_prob_test")

        # Build tree with prompt + continuation
        # Prompt: "a", "b"
        # Continuation: "x" or "y"
        t1 = tree.root.add_child("a", logprob=-0.1, token_id=1)
        t2 = t1.add_child("b", logprob=-0.2, token_id=2)

        # Continuations after prompt
        tx = t2.add_child("x", logprob=-0.3, token_id=3)
        tx.mark_as_trajectory()

        ty = t2.add_child("y", logprob=-0.5, token_id=4)
        ty.mark_as_trajectory()

        prompt = String(tokens=("a", "b"))
        traj_nodes = tree.root.get_trajectory_nodes()

        probs = tree.root.get_conditional_probabilities(
            traj_nodes, prompt, normalize=True
        )

        # Should sum to 1 when normalized
        assert np.sum(probs) == pytest.approx(1.0)

        # "x" should have higher probability (less negative logprob)
        assert probs[0] > probs[1]  # Assuming tx is first


class TestCompositeStructures:
    """Test composite structure integration."""

    def test_composite_in_system(self):
        """Test using CompositeStructure in VectorSystem."""

        def length_fn(s: String) -> float:
            return min(1.0, len(s.tokens) / 10)

        def vowel_fn(s: String) -> float:
            text = s.to_text().lower()
            if not text:
                return 0.5
            return sum(1 for c in text if c in "aeiou") / len(text)

        # Create composite structure
        composite = CompositeStructure(
            [
                FunctionalStructure(length_fn, name="length"),
                FunctionalStructure(vowel_fn, name="vowels"),
            ],
            combiner="mean",
            name="text_quality",
        )

        # Simple structure
        simple = FunctionalStructure(
            lambda s: 1.0 if len(s.tokens) > 0 else 0.0, name="non_empty"
        )

        # Create system with both
        system = VectorSystem([composite, simple])

        string = String(tokens=("hello", " ", "world"))
        compliance = system.compliance(string)

        assert len(compliance) == 2
        # All values should be in [0, 1]
        assert np.all(compliance.to_array() >= 0)
        assert np.all(compliance.to_array() <= 1)


class TestOperatorVariety:
    """Test various operator combinations."""

    @pytest.fixture
    def simple_system(self):
        """Create simple test system."""
        structures = [
            FunctionalStructure(lambda s: 0.5, name="s1"),
            FunctionalStructure(lambda s: 0.7, name="s2"),
        ]
        return VectorSystem(structures)

    def test_different_score_operators(self, simple_system):
        """Test different score operators give different results."""
        string = String(tokens=("test",))
        compliance = simple_system.compliance(string)

        # Test different operators
        l1_score = L1ScoreOperator()(compliance)
        l2_score = L2ScoreOperator()(compliance)
        mean_score = MeanScoreOperator()(compliance)

        # All should be in [0, 1] and finite
        for score in [l1_score, l2_score, mean_score]:
            assert 0 <= score <= 1
            assert np.isfinite(score)

    def test_different_difference_operators(self, simple_system):
        """Test different difference operators."""
        s1 = String(tokens=("hello",))
        s2 = String(tokens=("world",))

        c1 = simple_system.compliance(s1)
        c2 = simple_system.compliance(s2)

        l1_diff = L1DifferenceOperator()(c1, c2)
        l2_diff = L2DifferenceOperator()(c1, c2)

        # Both compliances are identical (fixed scores), so diff should be 0
        assert l1_diff == pytest.approx(0.0)
        assert l2_diff == pytest.approx(0.0)


class TestSyntheticDataFlow:
    """Test data flow with synthetic data."""

    def test_end_to_end_analysis(self):
        """Test complete analysis pipeline."""

        # 1. Define structures
        def sentiment_fn(s: String) -> float:
            text = s.to_text().lower()
            positive = ["good", "great", "happy", "love"]
            negative = ["bad", "terrible", "sad", "hate"]
            pos_count = sum(1 for w in positive if w in text)
            neg_count = sum(1 for w in negative if w in text)
            if pos_count + neg_count == 0:
                return 0.5
            return pos_count / (pos_count + neg_count)

        def formality_fn(s: String) -> float:
            text = s.to_text()
            # Simple heuristic: capital letters = more formal
            if not text:
                return 0.5
            caps = sum(1 for c in text if c.isupper())
            return min(1.0, caps / max(1, len(text)) * 10)

        # 2. Create system
        system = VectorSystem(
            [
                FunctionalStructure(sentiment_fn, name="sentiment"),
                FunctionalStructure(formality_fn, name="formality"),
            ]
        )

        # 3. Create synthetic trajectories
        trajectories = [
            String.from_text("I love this great product"),
            String.from_text("This is terrible and bad"),
            String.from_text("The meeting was productive"),
        ]
        probs = np.array([0.4, 0.3, 0.3])

        # 4. Compute core
        core = compute_core_from_trajectories(system, trajectories, probs)

        # 5. Evaluate individual strings against core
        test_string = String.from_text("I really love happy things")
        compliance = system.compliance(test_string)

        orientation = VectorOrientation(compliance, core, system.difference_operator)

        # 6. Compute deviance
        deviance = orientation.deviance()

        # Verify results are sensible
        assert deviance >= 0
        assert np.isfinite(deviance)
