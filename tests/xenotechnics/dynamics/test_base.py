"""
Tests for abstract dynamics base classes.

Tests for xenotechnics/dynamics/base.py
"""

from __future__ import annotations

import numpy as np
import pytest

from xenotechnics.common import FunctionalStructure, String
from xenotechnics.dynamics.base import AbstractDynamics, DynamicsState
from xenotechnics.systems.vector_system import VectorSystem


class TestDynamicsState:
    """Test DynamicsState dataclass."""

    def test_creation(self):
        """Test creating DynamicsState."""
        state = DynamicsState(
            step=5,
            current_string=String(tokens=("hello", " ", "world")),
            x_phi=np.array([0.1, 0.2, 0.3]),
            y_phi=np.array([0.4, 0.5, 0.6]),
            z_phi=np.array([0.7, 0.8, 0.9]),
        )

        assert state.step == 5
        assert len(state.current_string.tokens) == 3
        assert len(state.x_phi) == 3
        assert len(state.y_phi) == 3
        assert len(state.z_phi) == 3

    def test_repr(self):
        """Test string representation."""
        state = DynamicsState(
            step=3,
            current_string=String(tokens=("test",)),
            x_phi=np.array([0.5, 0.5]),
            y_phi=np.array([0.3, 0.7]),
            z_phi=np.array([0.2, 0.8]),
        )

        repr_str = repr(state)
        assert "DynamicsState" in repr_str
        assert "step=3" in repr_str
        assert "x_ϕ" in repr_str or "x_phi" in repr_str

    def test_zero_vectors(self):
        """Test with zero vectors."""
        state = DynamicsState(
            step=0,
            current_string=String.empty(),
            x_phi=np.zeros(3),
            y_phi=np.zeros(3),
            z_phi=np.zeros(3),
        )

        assert state.step == 0
        assert np.linalg.norm(state.x_phi) == 0.0
        assert np.linalg.norm(state.y_phi) == 0.0
        assert np.linalg.norm(state.z_phi) == 0.0


class ConcreteDynamics(AbstractDynamics):
    """Concrete implementation of AbstractDynamics for testing."""

    def compute_state(
        self,
        step: int,
        current_string: String,
        trajectory: String,
        root_core: np.ndarray,
        continuation_core: np.ndarray,
    ) -> DynamicsState:
        """Simple implementation that returns mock values."""
        # Compute mock orientations
        compliance = self.system.compliance(current_string).to_array()
        traj_compliance = self.system.compliance(trajectory).to_array()

        # x_phi: expected compliance of continuations
        x_phi = continuation_core

        # y_phi: orientation from root
        y_phi = compliance - root_core

        # z_phi: orientation to trajectory end
        z_phi = traj_compliance - compliance

        return DynamicsState(
            step=step,
            current_string=current_string,
            x_phi=x_phi,
            y_phi=y_phi,
            z_phi=z_phi,
        )


class TestAbstractDynamics:
    """Test AbstractDynamics class."""

    @pytest.fixture
    def simple_system(self):
        """Create simple system for testing."""

        def length_fn(s: String) -> float:
            return min(1.0, len(s.tokens) / 10)

        def char_fn(s: String) -> float:
            text = s.to_text()
            return min(1.0, len(text) / 50) if text else 0.0

        structures = [
            FunctionalStructure(length_fn, name="length"),
            FunctionalStructure(char_fn, name="chars"),
        ]
        return VectorSystem(structures)

    @pytest.fixture
    def dynamics(self, simple_system):
        """Create dynamics tracker for testing."""
        return ConcreteDynamics(simple_system)

    def test_init(self, simple_system):
        """Test initialization."""
        dynamics = ConcreteDynamics(simple_system)

        assert dynamics.system is simple_system
        assert dynamics.states == []

    def test_track_trajectory_simple(self, dynamics):
        """Test tracking a simple trajectory."""
        trajectory = String(tokens=("hello", " ", "world"))
        root_core = np.array([0.0, 0.0])
        continuation_cores = [
            np.array([0.1, 0.02]),
            np.array([0.2, 0.04]),
            np.array([0.3, 0.1]),
        ]

        dynamics.track_trajectory(trajectory, root_core, continuation_cores)

        assert len(dynamics) == 3
        assert len(dynamics.states) == 3

    def test_track_trajectory_clears_previous(self, dynamics):
        """Test that tracking clears previous states."""
        # First trajectory
        traj1 = String(tokens=("first",))
        dynamics.track_trajectory(traj1, np.zeros(2), [np.zeros(2)])
        assert len(dynamics) == 1

        # Second trajectory should clear first
        traj2 = String(tokens=("second", "trajectory", "here"))
        dynamics.track_trajectory(
            traj2, np.zeros(2), [np.zeros(2), np.zeros(2), np.zeros(2)]
        )
        assert len(dynamics) == 3

    def test_get_evolution(self, dynamics):
        """Test get_evolution method."""
        trajectory = String(tokens=("a", "b", "c"))
        root_core = np.array([0.0, 0.0])
        continuation_cores = [
            np.array([0.1, 0.1]),
            np.array([0.2, 0.2]),
            np.array([0.3, 0.3]),
        ]

        dynamics.track_trajectory(trajectory, root_core, continuation_cores)
        x_phis, y_phis, z_phis = dynamics.get_evolution()

        assert x_phis.shape == (3, 2)
        assert y_phis.shape == (3, 2)
        assert z_phis.shape == (3, 2)

    def test_get_evolution_empty(self, dynamics):
        """Test get_evolution with no tracked states."""
        x_phis, y_phis, z_phis = dynamics.get_evolution()

        assert len(x_phis) == 0
        assert len(y_phis) == 0
        assert len(z_phis) == 0

    def test_len(self, dynamics):
        """Test __len__ returns number of states."""
        assert len(dynamics) == 0

        trajectory = String(tokens=("a", "b"))
        dynamics.track_trajectory(trajectory, np.zeros(2), [np.zeros(2), np.zeros(2)])
        assert len(dynamics) == 2

    def test_repr(self, dynamics):
        """Test string representation."""
        repr_str = repr(dynamics)
        assert "ConcreteDynamics" in repr_str
        assert "0 states" in repr_str

        trajectory = String(tokens=("test",))
        dynamics.track_trajectory(trajectory, np.zeros(2), [np.zeros(2)])
        repr_str = repr(dynamics)
        assert "1 states" in repr_str

    def test_state_step_indices(self, dynamics):
        """Test that state step indices are correct."""
        trajectory = String(tokens=("a", "b", "c", "d"))
        root_core = np.zeros(2)
        continuation_cores = [np.zeros(2) for _ in range(4)]

        dynamics.track_trajectory(trajectory, root_core, continuation_cores)

        for i, state in enumerate(dynamics.states):
            assert state.step == i

    def test_current_string_progression(self, dynamics):
        """Test that current_string grows with each step."""
        trajectory = String(tokens=("hello", " ", "world"))
        root_core = np.zeros(2)
        continuation_cores = [np.zeros(2) for _ in range(3)]

        dynamics.track_trajectory(trajectory, root_core, continuation_cores)

        # Each state should have progressively longer string
        for i, state in enumerate(dynamics.states):
            expected_tokens = trajectory.tokens[: i + 1]
            assert state.current_string.tokens == expected_tokens

    def test_continuation_core_fallback(self, dynamics):
        """Test fallback when fewer continuation cores than trajectory length."""
        trajectory = String(tokens=("a", "b", "c", "d"))
        root_core = np.zeros(2)
        # Only 2 cores for 4-token trajectory
        continuation_cores = [np.array([0.1, 0.1]), np.array([0.2, 0.2])]

        dynamics.track_trajectory(trajectory, root_core, continuation_cores)

        # Should still track all 4 steps
        assert len(dynamics) == 4
