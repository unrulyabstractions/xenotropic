"""
Tests for linear dynamics.

Tests for xenotechnics/dynamics/linear.py
"""

from __future__ import annotations

import numpy as np
import pytest

from xenotechnics.common import FunctionalStructure, String
from xenotechnics.systems.vector_system import VectorSystem


class TestLinearDynamics:
    """Test LinearDynamics class."""

    @pytest.fixture
    def simple_system(self):
        """Create simple system for dynamics testing."""

        def length_fn(s: String) -> float:
            return min(1.0, len(s) / 10)

        def vowel_fn(s: String) -> float:
            text = s.to_text().lower()
            if not text:
                return 0.0
            vowels = sum(1 for c in text if c in "aeiou")
            return vowels / max(1, len(text))

        structures = [
            FunctionalStructure(length_fn, name="length"),
            FunctionalStructure(vowel_fn, name="vowels"),
        ]
        return VectorSystem(structures)

    def test_dynamics_import(self):
        """Test LinearDynamics can be imported."""
        from xenotechnics.dynamics.linear import LinearDynamics

        assert LinearDynamics is not None

    def test_dynamics_state_import(self):
        """Test DynamicsState can be imported."""
        from xenotechnics.dynamics.base import DynamicsState

        assert DynamicsState is not None

    def test_dynamics_computation(self, simple_system):
        """Test basic dynamics computation."""
        from xenotechnics.dynamics.linear import LinearDynamics

        dynamics = LinearDynamics(simple_system)

        current_string = String(tokens=("hello",))
        trajectory = String(tokens=("hello", " ", "world"))
        root_core = np.array([0.3, 0.4])
        continuation_core = np.array([0.5, 0.5])

        state = dynamics.compute_state(
            step=1,
            current_string=current_string,
            trajectory=trajectory,
            root_core=root_core,
            continuation_core=continuation_core,
        )

        assert state.step == 1
        assert state.current_string == current_string
        assert state.x_phi is not None
        assert state.y_phi is not None
        assert state.z_phi is not None

    def test_dynamics_state_components(self, simple_system):
        """Test dynamics state has correct shape."""
        from xenotechnics.dynamics.linear import LinearDynamics

        dynamics = LinearDynamics(simple_system)

        current_string = String(tokens=("a", "b"))
        trajectory = String(tokens=("a", "b", "c", "d"))
        root_core = np.array([0.3, 0.4])
        continuation_core = np.array([0.5, 0.5])

        state = dynamics.compute_state(
            step=0,
            current_string=current_string,
            trajectory=trajectory,
            root_core=root_core,
            continuation_core=continuation_core,
        )

        # x_phi = continuation_core
        np.testing.assert_array_equal(state.x_phi, continuation_core)

        # y_phi and z_phi should have same dimension as system
        assert len(state.y_phi) == len(simple_system)
        assert len(state.z_phi) == len(simple_system)


class TestDynamicsState:
    """Test DynamicsState dataclass."""

    def test_creation(self):
        """Test creating DynamicsState."""
        from xenotechnics.dynamics.base import DynamicsState

        state = DynamicsState(
            step=0,
            current_string=String(tokens=("test",)),
            x_phi=np.array([0.5, 0.5]),
            y_phi=np.array([0.1, 0.2]),
            z_phi=np.array([0.3, 0.4]),
        )

        assert state.step == 0
        assert len(state.x_phi) == 2
        assert len(state.y_phi) == 2
        assert len(state.z_phi) == 2

    def test_optional_fields(self):
        """Test DynamicsState with optional fields."""
        from xenotechnics.dynamics.base import DynamicsState

        # Minimal state
        state = DynamicsState(
            step=0,
            current_string=String.empty(),
            x_phi=np.array([0.5]),
            y_phi=np.array([0.1]),
            z_phi=np.array([0.3]),
        )

        assert state.step == 0
