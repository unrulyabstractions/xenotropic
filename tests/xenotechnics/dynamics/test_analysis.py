"""
Tests for dynamics analysis functions.

Tests for xenotechnics/dynamics/analysis.py
"""

from __future__ import annotations

import numpy as np
import pytest

from xenotechnics.common import String
from xenotechnics.dynamics.analysis import (
    analyze_evolution,
    compute_trajectory_stability,
    identify_critical_steps,
)
from xenotechnics.dynamics.base import AbstractDynamics, DynamicsState


class ConcreteDynamics(AbstractDynamics):
    """Concrete implementation for testing."""

    def compute_state(
        self,
        step: int,
        current_string: String,
        trajectory: String,
        root_core: np.ndarray,
        continuation_core: np.ndarray,
    ) -> DynamicsState:
        """Simple implementation."""
        return DynamicsState(
            step=step,
            current_string=current_string,
            x_phi=continuation_core,
            y_phi=np.random.randn(len(continuation_core)),
            z_phi=np.random.randn(len(continuation_core)),
        )


class MockDynamics:
    """Mock dynamics with controlled evolution."""

    def __init__(self, x_phis, y_phis, z_phis):
        self.states = [
            type("State", (), {"x_phi": x, "y_phi": y, "z_phi": z})()
            for x, y, z in zip(x_phis, y_phis, z_phis)
        ]

    def get_evolution(self):
        x = np.array([s.x_phi for s in self.states])
        y = np.array([s.y_phi for s in self.states])
        z = np.array([s.z_phi for s in self.states])
        return x, y, z


class TestAnalyzeEvolution:
    """Test analyze_evolution function."""

    def test_empty_dynamics(self):
        """Test with no states."""
        dynamics = MockDynamics([], [], [])
        result = analyze_evolution(dynamics)
        assert result == {}

    def test_single_state(self):
        """Test with single state."""
        dynamics = MockDynamics(
            [np.array([0.5, 0.5])],
            [np.array([0.1, 0.2])],
            [np.array([0.3, 0.4])],
        )
        result = analyze_evolution(dynamics)

        assert result["num_steps"] == 1
        assert "x_phi_stats" in result
        assert "y_phi_stats" in result
        assert "z_phi_stats" in result

    def test_multiple_states(self):
        """Test with multiple states."""
        dynamics = MockDynamics(
            [
                np.array([0.1, 0.1]),
                np.array([0.2, 0.2]),
                np.array([0.3, 0.3]),
            ],
            [
                np.array([0.0, 0.0]),
                np.array([0.1, 0.1]),
                np.array([0.2, 0.2]),
            ],
            [
                np.array([1.0, 1.0]),
                np.array([0.5, 0.5]),
                np.array([0.1, 0.1]),
            ],
        )
        result = analyze_evolution(dynamics)

        assert result["num_steps"] == 3

        # Check x_phi stats
        x_stats = result["x_phi_stats"]
        assert x_stats["start"] == pytest.approx(np.linalg.norm([0.1, 0.1]))
        assert x_stats["end"] == pytest.approx(np.linalg.norm([0.3, 0.3]))
        assert "mean" in x_stats
        assert "std" in x_stats
        assert "max" in x_stats
        assert "min" in x_stats

    def test_stats_computed_correctly(self):
        """Test that stats are computed correctly."""
        # Create predictable dynamics
        x_vals = [
            np.array([1.0, 0.0]),  # norm = 1.0
            np.array([2.0, 0.0]),  # norm = 2.0
            np.array([3.0, 0.0]),  # norm = 3.0
        ]
        dynamics = MockDynamics(
            x_vals,
            [np.zeros(2)] * 3,
            [np.zeros(2)] * 3,
        )
        result = analyze_evolution(dynamics)

        x_stats = result["x_phi_stats"]
        assert x_stats["start"] == pytest.approx(1.0)
        assert x_stats["end"] == pytest.approx(3.0)
        assert x_stats["mean"] == pytest.approx(2.0)
        assert x_stats["min"] == pytest.approx(1.0)
        assert x_stats["max"] == pytest.approx(3.0)


class TestIdentifyCriticalSteps:
    """Test identify_critical_steps function."""

    def test_empty_dynamics(self):
        """Test with no states."""
        dynamics = MockDynamics([], [], [])
        result = identify_critical_steps(dynamics)
        assert result == []

    def test_single_state(self):
        """Test with single state."""
        dynamics = MockDynamics(
            [np.array([0.5])],
            [np.array([0.5])],
            [np.array([0.5])],
        )
        result = identify_critical_steps(dynamics)
        assert result == []

    def test_no_critical_steps(self):
        """Test with small changes below threshold."""
        dynamics = MockDynamics(
            [np.array([0.0])] * 3,
            [
                np.array([0.1]),
                np.array([0.11]),  # delta = 0.01
                np.array([0.12]),  # delta = 0.01
            ],
            [np.array([0.0])] * 3,
        )
        result = identify_critical_steps(dynamics, threshold=0.5, component="y_phi")
        assert result == []

    def test_identify_critical_steps(self):
        """Test identification of critical steps."""
        dynamics = MockDynamics(
            [np.array([0.0])] * 4,
            [
                np.array([0.0]),
                np.array([0.1]),  # delta = 0.1
                np.array([0.9]),  # delta = 0.8 - CRITICAL
                np.array([1.0]),  # delta = 0.1
            ],
            [np.array([0.0])] * 4,
        )
        result = identify_critical_steps(dynamics, threshold=0.5, component="y_phi")
        assert result == [2]  # Step 2 has large change

    def test_multiple_critical_steps(self):
        """Test multiple critical steps."""
        dynamics = MockDynamics(
            [np.array([0.0])] * 5,
            [
                np.array([0.0]),
                np.array([1.0]),  # delta = 1.0 - CRITICAL
                np.array([1.1]),  # delta = 0.1
                np.array([2.1]),  # delta = 1.0 - CRITICAL
                np.array([2.2]),  # delta = 0.1
            ],
            [np.array([0.0])] * 5,
        )
        result = identify_critical_steps(dynamics, threshold=0.5, component="y_phi")
        assert result == [1, 3]

    def test_different_components(self):
        """Test with different components."""
        dynamics = MockDynamics(
            [
                np.array([0.0]),
                np.array([1.0]),  # delta = 1.0 > 0.5, critical
                np.array([1.0]),
            ],
            [
                np.array([0.0]),
                np.array([0.1]),  # delta = 0.1 < 0.5, not critical
                np.array([0.1]),
            ],
            [
                np.array([0.0]),
                np.array([0.6]),  # delta = 0.6 > 0.5, critical
                np.array([0.6]),
            ],
        )

        x_critical = identify_critical_steps(dynamics, threshold=0.5, component="x_phi")
        y_critical = identify_critical_steps(dynamics, threshold=0.5, component="y_phi")
        z_critical = identify_critical_steps(dynamics, threshold=0.5, component="z_phi")

        assert x_critical == [1]
        assert y_critical == []
        assert z_critical == [1]

    def test_invalid_component_raises(self):
        """Test that invalid component raises error."""
        dynamics = MockDynamics(
            [np.array([0.0])] * 2,
            [np.array([0.0])] * 2,
            [np.array([0.0])] * 2,
        )
        with pytest.raises(ValueError, match="Unknown component"):
            identify_critical_steps(dynamics, component="invalid")


class TestComputeTrajectoryStability:
    """Test compute_trajectory_stability function."""

    def test_empty_dynamics(self):
        """Test with no states."""
        dynamics = MockDynamics([], [], [])
        result = compute_trajectory_stability(dynamics)
        assert result == 0.0

    def test_single_state(self):
        """Test with single state."""
        dynamics = MockDynamics(
            [np.array([0.0])],
            [np.array([0.0])],
            [np.array([0.5])],
        )
        result = compute_trajectory_stability(dynamics)
        assert result == 0.0

    def test_stable_trajectory(self):
        """Test trajectory with constant z_phi norms."""
        dynamics = MockDynamics(
            [np.array([0.0])] * 3,
            [np.array([0.0])] * 3,
            [
                np.array([1.0, 0.0]),  # norm = 1.0
                np.array([0.0, 1.0]),  # norm = 1.0
                np.array([0.707, 0.707]),  # norm ≈ 1.0
            ],
        )
        result = compute_trajectory_stability(dynamics)
        # Variance should be near zero since all norms are 1.0
        assert result == pytest.approx(0.0, abs=0.01)

    def test_unstable_trajectory(self):
        """Test trajectory with varying z_phi norms."""
        dynamics = MockDynamics(
            [np.array([0.0])] * 3,
            [np.array([0.0])] * 3,
            [
                np.array([0.0, 0.0]),  # norm = 0.0
                np.array([1.0, 0.0]),  # norm = 1.0
                np.array([2.0, 0.0]),  # norm = 2.0
            ],
        )
        result = compute_trajectory_stability(dynamics)
        # Variance should be positive since norms vary
        assert result > 0.0

    def test_decreasing_z_phi(self):
        """Test trajectory where z_phi decreases to zero."""
        dynamics = MockDynamics(
            [np.array([0.0])] * 4,
            [np.array([0.0])] * 4,
            [
                np.array([2.0, 0.0]),  # norm = 2.0
                np.array([1.0, 0.0]),  # norm = 1.0
                np.array([0.5, 0.0]),  # norm = 0.5
                np.array([0.1, 0.0]),  # norm = 0.1
            ],
        )
        result = compute_trajectory_stability(dynamics)
        # Variance of [2.0, 1.0, 0.5, 0.1] = Var([2, 1, 0.5, 0.1])
        expected_var = np.var([2.0, 1.0, 0.5, 0.1])
        assert result == pytest.approx(expected_var)
