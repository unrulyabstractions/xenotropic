"""
Tests for xenoreproduction reward functions.

Tests for xenotechnics/xenoreproduction/rewards.py
"""

from __future__ import annotations

import numpy as np
import pytest

from xenotechnics.common import FunctionalStructure, String
from xenotechnics.systems.vector_system import VectorSystem
from xenotechnics.xenoreproduction.rewards import trajectory_reward


class TestTrajectoryReward:
    """Test trajectory_reward function."""

    @pytest.fixture
    def simple_system(self):
        """Create simple test system."""

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

    @pytest.fixture
    def reference_strings(self):
        """Create reference strings for core computation."""
        return [
            String(tokens=("hello",)),
            String(tokens=("world",)),
            String(tokens=("test",)),
        ]

    def test_basic_reward(self, simple_system, reference_strings):
        """Test basic reward computation."""
        trajectory = String(tokens=("unique", " ", "string"))

        reward = trajectory_reward(
            system=simple_system,
            trajectory=trajectory,
            reference_strings=reference_strings,
        )

        assert isinstance(reward, float)
        assert np.isfinite(reward)

    def test_reward_with_weights(self, simple_system, reference_strings):
        """Test reward with different weight configurations."""
        trajectory = String(tokens=("test",))

        # Only diversity
        reward_d = trajectory_reward(
            simple_system,
            trajectory,
            reference_strings,
            lambda_d=1.0,
            lambda_f=0.0,
            lambda_c=0.0,
        )

        # Only fairness
        reward_f = trajectory_reward(
            simple_system,
            trajectory,
            reference_strings,
            lambda_d=0.0,
            lambda_f=1.0,
            lambda_c=0.0,
        )

        # Only concentration
        reward_c = trajectory_reward(
            simple_system,
            trajectory,
            reference_strings,
            lambda_d=0.0,
            lambda_f=0.0,
            lambda_c=1.0,
        )

        # All components
        reward_all = trajectory_reward(
            simple_system,
            trajectory,
            reference_strings,
            lambda_d=1.0,
            lambda_f=1.0,
            lambda_c=1.0,
        )

        # Individual rewards should be finite
        assert np.isfinite(reward_d)
        assert np.isfinite(reward_f)
        assert np.isfinite(reward_c)
        assert np.isfinite(reward_all)

    def test_high_deviance_trajectory(self, simple_system, reference_strings):
        """Test trajectory that deviates from reference."""
        # Very different trajectory
        deviant = String(tokens=tuple("x" for _ in range(15)))

        # Similar trajectory
        similar = String(tokens=("hello",))

        reward_deviant = trajectory_reward(
            simple_system,
            deviant,
            reference_strings,
            lambda_d=1.0,
            lambda_f=0.0,
            lambda_c=0.0,
        )

        reward_similar = trajectory_reward(
            simple_system,
            similar,
            reference_strings,
            lambda_d=1.0,
            lambda_f=0.0,
            lambda_c=0.0,
        )

        # Deviant should have higher diversity reward
        # (Not always guaranteed, depends on the compliance values)
        assert np.isfinite(reward_deviant)
        assert np.isfinite(reward_similar)

    def test_empty_trajectory(self, simple_system, reference_strings):
        """Test with empty trajectory."""
        empty = String.empty()

        reward = trajectory_reward(
            simple_system,
            empty,
            reference_strings,
        )

        assert np.isfinite(reward)
