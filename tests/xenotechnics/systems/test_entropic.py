"""
Tests for entropic system implementations.

Tests for xenotechnics/systems/entropic.py
"""

from __future__ import annotations

import numpy as np
import pytest

from xenotechnics.common import FunctionalStructure, String
from xenotechnics.systems.entropic import DeficitSystem, ExcessSystem
from xenotechnics.systems.vector_system import VectorSystem


class TestExcessSystem:
    """Test ExcessSystem class."""

    @pytest.fixture
    def base_system(self):
        """Create base system for testing."""

        def length_fn(s: String) -> float:
            return min(1.0, len(s.tokens) / 10)

        def vowel_fn(s: String) -> float:
            text = s.to_text().lower()
            if not text:
                return 0.5
            return sum(1 for c in text if c in "aeiou") / max(1, len(text))

        structures = [
            FunctionalStructure(length_fn, name="length"),
            FunctionalStructure(vowel_fn, name="vowels"),
        ]
        return VectorSystem(structures)

    def test_init_default_baseline(self, base_system):
        """Test initialization with default uniform baseline."""
        excess_system = ExcessSystem(base_system)

        assert excess_system.n_structures == 2
        assert len(excess_system.baseline) == 2
        assert np.allclose(excess_system.baseline, [0.5, 0.5])

    def test_init_custom_baseline(self, base_system):
        """Test initialization with custom baseline."""
        baseline = np.array([0.3, 0.7])
        excess_system = ExcessSystem(base_system, baseline=baseline)

        assert np.allclose(excess_system.baseline, [0.3, 0.7])

    def test_init_invalid_baseline_length(self, base_system):
        """Test initialization with wrong baseline length raises error."""
        baseline = np.array([0.3, 0.7, 0.5])  # Wrong length

        with pytest.raises(ValueError, match="Baseline length"):
            ExcessSystem(base_system, baseline=baseline)

    def test_compliance_no_excess(self, base_system):
        """Test compliance when values are below baseline."""
        # With baseline at 0.5, a short string with few vowels should show no excess
        excess_system = ExcessSystem(base_system, baseline=np.array([0.8, 0.8]))

        s = String(tokens=("hi",))  # Short, low vowel ratio
        compliance = excess_system.compliance(s)

        # Values below baseline should give zero excess
        assert compliance.to_array()[0] == 0.0  # length below baseline

    def test_compliance_with_excess(self, base_system):
        """Test compliance when values exceed baseline."""
        excess_system = ExcessSystem(base_system, baseline=np.array([0.1, 0.1]))

        s = String(tokens=("hello", " ", "world", " ", "test"))
        compliance = excess_system.compliance(s)

        # Values above low baseline should show excess
        excess_vector = compliance.to_array()
        assert excess_vector[0] > 0  # Length exceeds baseline

    def test_structure_names(self, base_system):
        """Test structure names are prefixed with 'excess_'."""
        excess_system = ExcessSystem(base_system)
        names = excess_system.structure_names()

        assert names == ["excess_length", "excess_vowels"]

    def test_len(self, base_system):
        """Test __len__ returns correct count."""
        excess_system = ExcessSystem(base_system)
        assert len(excess_system) == 2

    def test_set_baseline(self, base_system):
        """Test set_baseline method."""
        excess_system = ExcessSystem(base_system)
        new_baseline = np.array([0.2, 0.8])
        excess_system.set_baseline(new_baseline)

        assert np.allclose(excess_system.baseline, [0.2, 0.8])

    def test_set_baseline_invalid_length(self, base_system):
        """Test set_baseline with wrong length raises error."""
        excess_system = ExcessSystem(base_system)
        new_baseline = np.array([0.2, 0.8, 0.5])

        with pytest.raises(ValueError, match="Baseline length"):
            excess_system.set_baseline(new_baseline)

    def test_repr(self, base_system):
        """Test string representation."""
        excess_system = ExcessSystem(base_system)
        repr_str = repr(excess_system)

        assert "ExcessSystem" in repr_str
        assert "2 structures" in repr_str

    def test_compute_core(self, base_system):
        """Test compute_core method."""
        excess_system = ExcessSystem(base_system, baseline=np.array([0.2, 0.2]))

        trajectories = [
            String(tokens=("hello",)),
            String(tokens=("world",)),
            String(tokens=("test",)),
        ]
        probabilities = np.array([0.5, 0.3, 0.2])

        core = excess_system.compute_core(trajectories, probabilities)
        assert core is not None


class TestDeficitSystem:
    """Test DeficitSystem class."""

    @pytest.fixture
    def base_system(self):
        """Create base system for testing."""

        def length_fn(s: String) -> float:
            return min(1.0, len(s.tokens) / 10)

        def vowel_fn(s: String) -> float:
            text = s.to_text().lower()
            if not text:
                return 0.5
            return sum(1 for c in text if c in "aeiou") / max(1, len(text))

        structures = [
            FunctionalStructure(length_fn, name="length"),
            FunctionalStructure(vowel_fn, name="vowels"),
        ]
        return VectorSystem(structures)

    def test_init_default_baseline(self, base_system):
        """Test initialization with default uniform baseline."""
        deficit_system = DeficitSystem(base_system)

        assert deficit_system.n_structures == 2
        assert len(deficit_system.baseline) == 2
        assert np.allclose(deficit_system.baseline, [0.5, 0.5])

    def test_init_custom_baseline(self, base_system):
        """Test initialization with custom baseline."""
        baseline = np.array([0.3, 0.7])
        deficit_system = DeficitSystem(base_system, baseline=baseline)

        assert np.allclose(deficit_system.baseline, [0.3, 0.7])

    def test_init_invalid_baseline_length(self, base_system):
        """Test initialization with wrong baseline length raises error."""
        baseline = np.array([0.3, 0.7, 0.5])  # Wrong length

        with pytest.raises(ValueError, match="Baseline length"):
            DeficitSystem(base_system, baseline=baseline)

    def test_compliance_no_deficit(self, base_system):
        """Test compliance when values are above baseline."""
        deficit_system = DeficitSystem(base_system, baseline=np.array([0.1, 0.1]))

        s = String(tokens=("hello", " ", "world", " ", "test"))
        compliance = deficit_system.compliance(s)

        # Values above baseline should give zero deficit
        deficit_vector = compliance.to_array()
        # Some values may be above baseline

    def test_compliance_with_deficit(self, base_system):
        """Test compliance when values are below baseline."""
        deficit_system = DeficitSystem(base_system, baseline=np.array([0.9, 0.9]))

        s = String(tokens=("hi",))  # Short string
        compliance = deficit_system.compliance(s)

        # Values below high baseline should show deficit
        deficit_vector = compliance.to_array()
        assert deficit_vector[0] > 0  # Length below baseline

    def test_structure_names(self, base_system):
        """Test structure names are prefixed with 'deficit_'."""
        deficit_system = DeficitSystem(base_system)
        names = deficit_system.structure_names()

        assert names == ["deficit_length", "deficit_vowels"]

    def test_len(self, base_system):
        """Test __len__ returns correct count."""
        deficit_system = DeficitSystem(base_system)
        assert len(deficit_system) == 2

    def test_set_baseline(self, base_system):
        """Test set_baseline method."""
        deficit_system = DeficitSystem(base_system)
        new_baseline = np.array([0.2, 0.8])
        deficit_system.set_baseline(new_baseline)

        assert np.allclose(deficit_system.baseline, [0.2, 0.8])

    def test_set_baseline_invalid_length(self, base_system):
        """Test set_baseline with wrong length raises error."""
        deficit_system = DeficitSystem(base_system)
        new_baseline = np.array([0.2, 0.8, 0.5])

        with pytest.raises(ValueError, match="Baseline length"):
            deficit_system.set_baseline(new_baseline)

    def test_repr(self, base_system):
        """Test string representation."""
        deficit_system = DeficitSystem(base_system)
        repr_str = repr(deficit_system)

        assert "DeficitSystem" in repr_str
        assert "2 structures" in repr_str

    def test_compute_core(self, base_system):
        """Test compute_core method."""
        deficit_system = DeficitSystem(base_system, baseline=np.array([0.8, 0.8]))

        trajectories = [
            String(tokens=("hello",)),
            String(tokens=("world",)),
            String(tokens=("test",)),
        ]
        probabilities = np.array([0.5, 0.3, 0.2])

        core = deficit_system.compute_core(trajectories, probabilities)
        assert core is not None


class TestExcessDeficitComplementary:
    """Test that Excess and Deficit systems are complementary."""

    @pytest.fixture
    def base_system(self):
        """Create base system for testing."""

        def simple_fn(s: String) -> float:
            return min(1.0, len(s.tokens) / 5)

        return VectorSystem([FunctionalStructure(simple_fn, name="simple")])

    def test_excess_plus_deficit_covers_deviation(self, base_system):
        """Test that excess + deficit = |deviation| from baseline."""
        baseline = np.array([0.5])

        excess_system = ExcessSystem(base_system, baseline=baseline)
        deficit_system = DeficitSystem(base_system, baseline=baseline)

        test_strings = [
            String(tokens=("a",)),
            String(tokens=("a", "b", "c")),
            String(tokens=("a", "b", "c", "d", "e")),
            String(tokens=("a", "b", "c", "d", "e", "f", "g")),
        ]

        for s in test_strings:
            base_comp = base_system.compliance(s).to_array()
            excess = excess_system.compliance(s).to_array()
            deficit = deficit_system.compliance(s).to_array()

            # excess + deficit should equal absolute deviation
            total = excess + deficit
            deviation = np.abs(base_comp - baseline)
            assert np.allclose(total, deviation)
