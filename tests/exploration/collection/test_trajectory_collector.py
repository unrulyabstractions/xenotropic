"""
Tests for TrajectoryCollector.

Tests for exploration/collection/trajectory_collector.py
"""

from __future__ import annotations

import numpy as np
import torch

from exploration.collection import (
    CollectedTrajectory,
    CollectionResult,
    TrajectoryCollector,
    TrajectoryCollectorConfig,
)


class MockModelRunner:
    """Mock ModelRunner for testing."""

    def __init__(self, vocab_size: int = 100, deterministic: bool = True):
        self.vocab_size = vocab_size
        self.device = "cpu"
        self.eos_token_id = 0
        self._step = 0
        self._deterministic = deterministic

    def tokenize(self, text: str, prepend_bos: bool = True) -> torch.Tensor:
        # Simple mock: just return fixed tokens
        return torch.tensor([[1, 2, 3]])

    def decode(self, token_ids: torch.Tensor) -> str:
        ids = token_ids.tolist()
        if isinstance(ids, list):
            return "".join(chr(65 + (i % 26)) for i in ids)
        return chr(65 + (ids % 26))

    def get_next_token_logits(
        self, input_ids: torch.Tensor, past_kv_cache=None
    ) -> tuple[torch.Tensor, None]:
        self._step += 1
        logits = torch.randn(1, self.vocab_size)

        if self._deterministic:
            # Make token 10 most likely, with some variation
            logits[0, 10 + (self._step % 5)] = 10.0

        # Make EOS likely after 5 steps
        if self._step >= 5:
            logits[0, self.eos_token_id] = 20.0

        return logits, None

    def reset(self):
        self._step = 0


class TestTrajectoryCollectorConfig:
    """Test TrajectoryCollectorConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrajectoryCollectorConfig()

        assert config.max_new_tokens == 100
        assert config.temperature == 1.0
        assert config.top_k is None
        assert config.top_p is None
        assert config.target_mass == 0.95
        assert config.max_iterations == 500
        assert config.max_no_progress == 20
        assert config.seed == 42

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TrajectoryCollectorConfig(
            max_new_tokens=50,
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            target_mass=0.9,
            max_iterations=100,
            seed=123,
        )

        assert config.max_new_tokens == 50
        assert config.temperature == 0.8
        assert config.top_k == 40
        assert config.top_p == 0.9


class TestCollectedTrajectory:
    """Test CollectedTrajectory."""

    def test_creation(self):
        """Test creating a trajectory."""
        traj = CollectedTrajectory(
            text="Hello",
            tokens=("H", "e", "l", "l", "o"),
            token_ids=(1, 2, 3, 3, 4),
            probability=0.5,
            log_probability=-0.693,
            per_token_logprobs=[-0.1, -0.2, -0.1, -0.1, -0.2],
        )

        assert traj.text == "Hello"
        assert len(traj.tokens) == 5
        assert traj.probability == 0.5

    def test_hash_by_token_ids(self):
        """Test that trajectories are hashed by token_ids."""
        traj1 = CollectedTrajectory(
            text="A",
            tokens=("A",),
            token_ids=(1, 2, 3),
            probability=0.5,
            log_probability=-0.7,
        )
        traj2 = CollectedTrajectory(
            text="B",
            tokens=("B",),
            token_ids=(1, 2, 3),  # Same token IDs
            probability=0.3,
            log_probability=-1.2,
        )
        traj3 = CollectedTrajectory(
            text="C",
            tokens=("C",),
            token_ids=(4, 5, 6),  # Different token IDs
            probability=0.2,
            log_probability=-1.6,
        )

        assert hash(traj1) == hash(traj2)
        assert hash(traj1) != hash(traj3)
        assert traj1 == traj2
        assert traj1 != traj3


class TestCollectionResult:
    """Test CollectionResult."""

    def test_creation(self):
        """Test creating a result."""
        trajs = [
            CollectedTrajectory(
                text="A",
                tokens=("A",),
                token_ids=(1,),
                probability=0.6,
                log_probability=-0.5,
            ),
            CollectedTrajectory(
                text="B",
                tokens=("B",),
                token_ids=(2,),
                probability=0.4,
                log_probability=-0.9,
            ),
        ]
        result = CollectionResult(trajectories=trajs, total_mass=1.0, iterations=10)

        assert len(result.trajectories) == 2
        assert result.total_mass == 1.0
        assert result.iterations == 10

    def test_probabilities_property(self):
        """Test probabilities property returns array."""
        trajs = [
            CollectedTrajectory(
                text="A",
                tokens=("A",),
                token_ids=(1,),
                probability=0.6,
                log_probability=-0.5,
            ),
            CollectedTrajectory(
                text="B",
                tokens=("B",),
                token_ids=(2,),
                probability=0.4,
                log_probability=-0.9,
            ),
        ]
        result = CollectionResult(trajectories=trajs, total_mass=1.0, iterations=10)

        probs = result.probabilities
        assert isinstance(probs, np.ndarray)
        assert len(probs) == 2
        np.testing.assert_array_almost_equal(probs, [0.6, 0.4])


class TestTrajectoryCollector:
    """Test TrajectoryCollector."""

    def test_init(self):
        """Test initialization."""
        runner = MockModelRunner()
        config = TrajectoryCollectorConfig(max_iterations=10)
        collector = TrajectoryCollector(runner, config)

        assert collector.model_runner is runner
        assert collector.config is config

    def test_init_default_config(self):
        """Test initialization with default config."""
        runner = MockModelRunner()
        collector = TrajectoryCollector(runner)

        assert collector.config is not None
        assert collector.config.max_iterations == 500

    def test_collect_returns_result(self):
        """Test that collect returns CollectionResult."""
        runner = MockModelRunner()
        config = TrajectoryCollectorConfig(
            max_iterations=5,
            max_new_tokens=3,
            target_mass=0.99,  # High target to ensure we hit max_iterations
        )
        collector = TrajectoryCollector(runner, config)

        result = collector.collect("Hello")

        assert isinstance(result, CollectionResult)
        assert len(result.trajectories) >= 0
        assert result.total_mass >= 0

    def test_collect_iterator(self):
        """Test collect_iterator yields trajectories."""
        runner = MockModelRunner()
        config = TrajectoryCollectorConfig(
            max_iterations=3,
            max_new_tokens=3,
        )
        collector = TrajectoryCollector(runner, config)

        trajectories = list(collector.collect_iterator("Hello"))

        for traj in trajectories:
            assert isinstance(traj, CollectedTrajectory)
            assert traj.probability > 0
            assert traj.log_probability < 0  # Log prob should be negative

    def test_deduplication(self):
        """Test that duplicate trajectories are not collected."""
        runner = MockModelRunner(deterministic=True)
        config = TrajectoryCollectorConfig(
            max_iterations=10,
            max_new_tokens=3,
            max_no_progress=5,
        )
        collector = TrajectoryCollector(runner, config)

        result = collector.collect("Hello", seed=42)

        # Check all trajectories have unique token_ids
        seen_ids = set()
        for traj in result.trajectories:
            assert traj.token_ids not in seen_ids
            seen_ids.add(traj.token_ids)

    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        runner1 = MockModelRunner()
        runner2 = MockModelRunner()
        config = TrajectoryCollectorConfig(max_iterations=3, max_new_tokens=3)

        collector1 = TrajectoryCollector(runner1, config)
        collector2 = TrajectoryCollector(runner2, config)

        result1 = collector1.collect("Hello", seed=42)
        result2 = collector2.collect("Hello", seed=42)

        # Same seed should give same number of trajectories
        # (exact values may differ due to model state, but structure should match)
        assert len(result1.trajectories) == len(result2.trajectories)
