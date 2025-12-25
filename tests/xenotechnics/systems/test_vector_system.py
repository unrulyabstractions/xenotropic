"""
Tests for VectorSystem and VectorSystemCompliance.

Tests for xenotechnics/systems/vector_system.py
"""

from __future__ import annotations

import numpy as np
import pytest

from xenotechnics.common import AbstractStructure, FunctionalStructure, String
from xenotechnics.systems.vector_system import (
    VectorOrientation,
    VectorSystem,
    VectorSystemCompliance,
    compute_core_from_trajectories,
    core_entropy,
)


class FixedScoreStructure(AbstractStructure):
    """Structure that returns a fixed score."""

    def __init__(self, score: float, name: str = "fixed"):
        super().__init__(name, f"Fixed score: {score}")
        self.score = score

    def compliance(self, string: String) -> float:
        return self.score


class TestVectorSystemCompliance:
    """Test VectorSystemCompliance class."""

    @pytest.fixture
    def simple_system(self):
        """Create simple 3-structure system."""
        structures = [
            FixedScoreStructure(0.5, "s1"),
            FixedScoreStructure(0.7, "s2"),
            FixedScoreStructure(0.3, "s3"),
        ]
        return VectorSystem(structures)

    def test_creation(self, simple_system):
        """Test creating VectorSystemCompliance."""
        vector = np.array([0.5, 0.7, 0.3])
        compliance = VectorSystemCompliance(
            system=simple_system, compliance_vector=vector, string=String.empty()
        )

        assert len(compliance) == 3
        np.testing.assert_array_equal(compliance.to_array(), vector)

    def test_vector_length_mismatch(self, simple_system):
        """Test error when vector length doesn't match system."""
        vector = np.array([0.5, 0.7])  # Wrong length
        with pytest.raises(ValueError, match="must match system size"):
            VectorSystemCompliance(
                system=simple_system, compliance_vector=vector, string=String.empty()
            )

    def test_to_array_returns_copy(self, simple_system):
        """Test to_array returns a copy, not the original."""
        vector = np.array([0.5, 0.7, 0.3])
        compliance = VectorSystemCompliance(
            system=simple_system, compliance_vector=vector
        )

        result = compliance.to_array()
        result[0] = 999  # Modify the result

        # Original should be unchanged
        assert compliance.vector[0] == 0.5

    def test_repr(self, simple_system):
        """Test __repr__ format."""
        vector = np.array([0.5, 0.7, 0.3])
        compliance = VectorSystemCompliance(
            system=simple_system, compliance_vector=vector
        )
        rep = repr(compliance)

        assert "VectorSystemCompliance" in rep
        assert "n=3" in rep
        assert "mean=" in rep


class TestVectorSystem:
    """Test VectorSystem class."""

    def test_creation_with_structures(self):
        """Test creating VectorSystem with structures."""
        structures = [
            FixedScoreStructure(0.5, "s1"),
            FixedScoreStructure(0.7, "s2"),
        ]
        system = VectorSystem(structures)

        assert len(system) == 2
        assert system.structure_names() == ["s1", "s2"]

    def test_empty_structures_raises(self):
        """Test error when no structures provided."""
        with pytest.raises(ValueError, match="at least one structure"):
            VectorSystem([])

    def test_compliance_computation(self):
        """Test compliance computation returns correct values."""
        structures = [
            FixedScoreStructure(0.5, "s1"),
            FixedScoreStructure(0.8, "s2"),
            FixedScoreStructure(0.3, "s3"),
        ]
        system = VectorSystem(structures)

        compliance = system.compliance(String.empty())

        assert isinstance(compliance, VectorSystemCompliance)
        expected = np.array([0.5, 0.8, 0.3])
        np.testing.assert_array_almost_equal(compliance.to_array(), expected)

    def test_compliance_with_functional_structure(self):
        """Test with FunctionalStructure."""

        def length_fn(s: String) -> float:
            return min(1.0, len(s) / 10)

        structures = [
            FunctionalStructure(length_fn, name="length"),
            FixedScoreStructure(0.5, "fixed"),
        ]
        system = VectorSystem(structures)

        # Short string
        short = String(tokens=("a", "b"))
        compliance = system.compliance(short)
        assert compliance.to_array()[0] == pytest.approx(0.2)  # 2/10
        assert compliance.to_array()[1] == pytest.approx(0.5)

        # Longer string
        long = String(tokens=tuple("x" for _ in range(15)))
        compliance = system.compliance(long)
        assert compliance.to_array()[0] == pytest.approx(1.0)  # Capped at 1

    def test_default_operators(self):
        """Test default operators are L2Squared."""
        from xenotechnics.operators import (
            L2SquaredDifferenceOperator,
            L2SquaredScoreOperator,
        )

        structures = [FixedScoreStructure(0.5, "s1")]
        system = VectorSystem(structures)

        assert isinstance(system.score_operator, L2SquaredScoreOperator)
        assert isinstance(system.difference_operator, L2SquaredDifferenceOperator)

    def test_custom_operators(self):
        """Test custom operators can be provided."""
        from xenotechnics.operators import L1DifferenceOperator, L1ScoreOperator

        structures = [FixedScoreStructure(0.5, "s1")]
        system = VectorSystem(
            structures,
            score_operator=L1ScoreOperator(),
            difference_operator=L1DifferenceOperator(),
        )

        assert isinstance(system.score_operator, L1ScoreOperator)
        assert isinstance(system.difference_operator, L1DifferenceOperator)

    def test_repr(self):
        """Test __repr__ format."""
        structures = [FixedScoreStructure(0.5, "s1"), FixedScoreStructure(0.7, "s2")]
        system = VectorSystem(structures)
        rep = repr(system)

        assert "VectorSystem" in rep
        assert "2 structures" in rep


class TestVectorOrientation:
    """Test VectorOrientation class."""

    @pytest.fixture
    def system_and_compliances(self):
        """Create system and two compliances."""
        structures = [
            FixedScoreStructure(0.5, "s1"),
            FixedScoreStructure(0.7, "s2"),
            FixedScoreStructure(0.3, "s3"),
        ]
        system = VectorSystem(structures)

        c1 = VectorSystemCompliance(
            system=system, compliance_vector=np.array([0.8, 0.6, 0.4])
        )
        c2 = VectorSystemCompliance(
            system=system, compliance_vector=np.array([0.5, 0.5, 0.5])
        )

        return system, c1, c2

    def test_creation(self, system_and_compliances):
        """Test creating VectorOrientation."""
        system, c1, c2 = system_and_compliances
        orientation = VectorOrientation(c1, c2, system.difference_operator)

        assert len(orientation) == 3
        expected = np.array([0.3, 0.1, -0.1])
        np.testing.assert_array_almost_equal(orientation.to_array(), expected)

    def test_deviance(self, system_and_compliances):
        """Test deviance computation."""
        system, c1, c2 = system_and_compliances
        orientation = VectorOrientation(c1, c2, system.difference_operator)

        deviance = orientation.deviance()
        assert deviance >= 0

    def test_to_array_returns_copy(self, system_and_compliances):
        """Test to_array returns a copy."""
        system, c1, c2 = system_and_compliances
        orientation = VectorOrientation(c1, c2, system.difference_operator)

        result = orientation.to_array()
        result[0] = 999

        assert orientation.vector[0] != 999

    def test_repr(self, system_and_compliances):
        """Test __repr__ format."""
        system, c1, c2 = system_and_compliances
        orientation = VectorOrientation(c1, c2, system.difference_operator)
        rep = repr(orientation)

        assert "VectorOrientation" in rep
        assert "||θ||=" in rep


class TestComputeCoreFromTrajectories:
    """Test compute_core_from_trajectories function."""

    @pytest.fixture
    def system(self):
        """Create system for core computation testing."""

        def length_fn(s: String) -> float:
            return min(1.0, len(s) / 10)

        def vowel_fn(s: String) -> float:
            text = s.to_text().lower()
            if not text:
                return 0.0
            vowels = sum(1 for c in text if c in "aeiou")
            return vowels / len(text)

        structures = [
            FunctionalStructure(length_fn, name="length"),
            FunctionalStructure(vowel_fn, name="vowels"),
        ]
        return VectorSystem(structures)

    def test_basic_core_computation(self, system):
        """Test basic core computation."""
        trajectories = [
            String(tokens=("hello",)),
            String(tokens=("world",)),
            String(tokens=("test",)),
        ]
        probs = np.array([0.5, 0.3, 0.2])

        core = compute_core_from_trajectories(system, trajectories, probs)

        assert isinstance(core, VectorSystemCompliance)
        assert core.string is None  # Core has no associated string
        assert len(core) == 2

    def test_uniform_probabilities(self, system):
        """Test with uniform probabilities."""
        trajectories = [
            String(tokens=("a", "b")),  # length: 0.2
            String(tokens=("c", "d")),  # length: 0.2
        ]
        probs = np.array([0.5, 0.5])

        core = compute_core_from_trajectories(system, trajectories, probs)

        # With uniform probs, core is just the mean
        # All have length 2, so length score = 0.2
        assert core.to_array()[0] == pytest.approx(0.2)

    def test_weighted_probabilities(self, system):
        """Test with weighted probabilities."""
        trajectories = [
            String(tokens=("a",)),  # length: 0.1
            String(tokens=("a", "b", "c", "d", "e")),  # length: 0.5
        ]
        probs = np.array([0.9, 0.1])

        core = compute_core_from_trajectories(system, trajectories, probs)

        # Core should be closer to first trajectory due to higher weight
        # Expected: 0.9 * 0.1 + 0.1 * 0.5 = 0.09 + 0.05 = 0.14
        assert core.to_array()[0] == pytest.approx(0.14)

    def test_empty_trajectories_raises(self, system):
        """Test error with empty trajectory list."""
        with pytest.raises(ValueError, match="Cannot compute core from empty"):
            compute_core_from_trajectories(system, [], np.array([]))

    def test_mismatched_lengths_raises(self, system):
        """Test error when trajectories and probabilities don't match."""
        trajectories = [String(tokens=("a",)), String(tokens=("b",))]
        probs = np.array([0.5])  # Wrong length

        with pytest.raises(ValueError, match="must match"):
            compute_core_from_trajectories(system, trajectories, probs)


class TestCoreEntropy:
    """Test core_entropy function."""

    def test_uniform_high_entropy(self):
        """Test uniform distribution has high entropy."""
        structures = [FixedScoreStructure(0.5, f"s{i}") for i in range(4)]
        system = VectorSystem(structures)

        # Uniform values
        compliance = VectorSystemCompliance(
            system=system, compliance_vector=np.array([0.25, 0.25, 0.25, 0.25])
        )

        entropy = core_entropy(compliance)
        # Max entropy for 4 elements = log(4) ≈ 1.386
        assert entropy == pytest.approx(np.log(4), rel=0.01)

    def test_peaked_low_entropy(self):
        """Test peaked distribution has low entropy."""
        structures = [FixedScoreStructure(0.5, f"s{i}") for i in range(4)]
        system = VectorSystem(structures)

        # Very peaked
        compliance = VectorSystemCompliance(
            system=system, compliance_vector=np.array([0.97, 0.01, 0.01, 0.01])
        )

        entropy = core_entropy(compliance)
        # Should be low
        assert entropy < 0.5
