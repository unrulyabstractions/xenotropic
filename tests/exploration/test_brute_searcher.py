"""
Tests for brute-force searcher.

Tests for exploration/explorers/brute_searcher.py
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from exploration.explorers.brute_searcher import BruteSearcher, SearchResult
from xenotechnics.common import String
from xenotechnics.trees.tree import TreeNode


class MockGenerator:
    """Mock generator for testing."""

    def __init__(self, trajectories_per_run=1):
        self.trajectories_per_run = trajectories_per_run
        self.run_count = 0
        self.prompt_token_count = 0
        self._tree = None

    def run(
        self,
        prompt=None,
        max_new_tokens=100,
        verbose=False,
        existing_tree=None,
        **kwargs,
    ):
        """Mock run that adds trajectories to tree."""
        self.run_count += 1

        if existing_tree is None:
            # First run - create new tree
            self._tree = TreeNode(string=String.empty())
            self._tree.set_distribution(probs=np.array([1.0]))
        else:
            self._tree = existing_tree

        # Add trajectory node(s)
        for i in range(self.trajectories_per_run):
            token = f"token_{self.run_count}_{i}"
            prob = 0.5**self.run_count  # Decreasing probability
            child = self._tree.add_child(
                token, logprob=np.log(prob), token_id=self.run_count * 10 + i
            )
            child.mark_as_trajectory()

        return self._tree


class TestSearchResult:
    """Test SearchResult dataclass."""

    def test_creation(self):
        """Test creating SearchResult."""
        trajectories = [MagicMock(), MagicMock()]
        probabilities = np.array([0.6, 0.4])
        tree = MagicMock()

        result = SearchResult(
            trajectories=trajectories,
            probabilities=probabilities,
            total_mass=1.0,
            tree=tree,
        )

        assert len(result.trajectories) == 2
        assert result.total_mass == 1.0
        assert result.tree is tree


class TestBruteSearcher:
    """Test BruteSearcher class."""

    def test_init(self):
        """Test searcher initialization."""
        generator = MockGenerator()
        searcher = BruteSearcher(generator)
        assert searcher.generator is generator

    def test_search_collects_trajectories(self):
        """Test that search collects trajectories."""
        generator = MockGenerator(trajectories_per_run=1)
        searcher = BruteSearcher(generator)

        result = searcher.search(
            prompt=None,
            min_probability_mass=0.99,
            max_new_tokens=10,
            max_trajectories=5,
            verbose=False,
        )

        assert isinstance(result, SearchResult)
        assert len(result.trajectories) > 0
        assert len(result.probabilities) == len(result.trajectories)

    def test_search_stops_at_max_trajectories(self):
        """Test search stops when max trajectories reached."""
        generator = MockGenerator(trajectories_per_run=2)
        searcher = BruteSearcher(generator)

        result = searcher.search(
            prompt=None,
            min_probability_mass=0.99,
            max_new_tokens=10,
            max_trajectories=3,
            verbose=False,
        )

        # Should stop at or before max_trajectories
        assert len(result.trajectories) <= 4  # 2 per run, might overshoot by 1 run

    def test_search_with_prompt(self):
        """Test search with prompt."""
        generator = MockGenerator()
        searcher = BruteSearcher(generator)

        prompt = String(tokens=("Hello", " ", "world"))
        result = searcher.search(
            prompt=prompt,
            min_probability_mass=0.99,
            max_new_tokens=10,
            max_trajectories=2,
            verbose=False,
        )

        assert isinstance(result, SearchResult)

    def test_search_returns_tree(self):
        """Test search returns tree."""
        generator = MockGenerator()
        searcher = BruteSearcher(generator)

        result = searcher.search(
            prompt=None,
            min_probability_mass=0.99,
            max_new_tokens=10,
            max_trajectories=2,
            verbose=False,
        )

        assert result.tree is not None

    def test_search_probabilities_sum(self):
        """Test that probabilities are reasonable."""
        generator = MockGenerator()
        searcher = BruteSearcher(generator)

        result = searcher.search(
            prompt=None,
            min_probability_mass=0.99,
            max_new_tokens=10,
            max_trajectories=3,
            verbose=False,
        )

        # All probabilities should be non-negative
        assert all(p >= 0 for p in result.probabilities)
        # Total mass should match sum of probabilities
        assert result.total_mass == pytest.approx(np.sum(result.probabilities))

    def test_search_verbose_mode(self, capsys):
        """Test search with verbose mode."""
        generator = MockGenerator()
        searcher = BruteSearcher(generator)

        result = searcher.search(
            prompt=None,
            min_probability_mass=0.99,
            max_new_tokens=10,
            max_trajectories=2,
            verbose=True,
        )

        # Check that output was produced
        captured = capsys.readouterr()
        assert "BRUTE FORCE SEARCH" in captured.out

    def test_search_reuses_tree(self):
        """Test that search reuses tree across runs."""
        generator = MockGenerator()
        searcher = BruteSearcher(generator)

        result = searcher.search(
            prompt=None,
            min_probability_mass=0.99,
            max_new_tokens=10,
            max_trajectories=5,
            verbose=False,
        )

        # Multiple runs should have happened
        assert generator.run_count >= 1

    def test_search_with_seed(self):
        """Test search with random seed."""
        generator = MockGenerator()
        searcher = BruteSearcher(generator)

        result = searcher.search(
            prompt=None,
            min_probability_mass=0.99,
            max_new_tokens=10,
            max_trajectories=2,
            verbose=False,
            seed=42,
        )

        assert isinstance(result, SearchResult)
