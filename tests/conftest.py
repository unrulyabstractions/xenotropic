"""
Pytest configuration and shared fixtures for xenotropic tests.

Provides synthetic data, mock models, and common test utilities.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from xenotechnics.common import String
from xenotechnics.trees.tree import TreeNode

# =============================================================================
# String Fixtures
# =============================================================================


@pytest.fixture
def empty_string() -> String:
    """Empty string fixture."""
    return String.empty()


@pytest.fixture
def simple_string() -> String:
    """Simple string with tokens."""
    return String(tokens=("Hello", " ", "world"))


@pytest.fixture
def string_with_prompt() -> String:
    """String with prompt portion."""
    return String(
        tokens=("The", " ", "cat", " ", "sat"),
        token_ids=(100, 200, 300, 200, 400),
        prompt_length=2,
    )


@pytest.fixture
def string_with_ids() -> String:
    """String with token IDs."""
    return String(
        tokens=("a", "b", "c"),
        token_ids=(1, 2, 3),
    )


# =============================================================================
# TreeNode Fixtures
# =============================================================================


@pytest.fixture
def root_node() -> TreeNode:
    """Root tree node with empty string."""
    return TreeNode(string=String.empty())


@pytest.fixture
def simple_tree() -> TreeNode:
    """Simple tree with a few nodes."""
    root = TreeNode(string=String.empty())

    # Add first level children
    child_a = root.add_child("a", logprob=-0.5, token_id=1)
    child_b = root.add_child("b", logprob=-1.0, token_id=2)

    # Add second level children
    child_a.add_child("x", logprob=-0.3, token_id=10)
    child_a.add_child("y", logprob=-0.7, token_id=11)
    child_b.add_child("z", logprob=-0.2, token_id=12)

    return root


@pytest.fixture
def tree_with_distribution() -> TreeNode:
    """Tree node with next-token distribution set."""
    node = TreeNode(string=String(tokens=("test",)))

    # Create synthetic distribution over 100 "vocabulary"
    probs = np.random.dirichlet(np.ones(100))
    node.set_distribution(probs=probs)

    return node


@pytest.fixture
def tree_with_trajectories() -> TreeNode:
    """Tree with marked trajectory nodes."""
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


# =============================================================================
# Mock Model Fixtures
# =============================================================================


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.vocab_size = 100
    tokenizer.eos_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.pad_token_id = 2

    # Simple encoding: character -> ASCII code
    def encode(text, **kwargs):
        return [ord(c) % 100 for c in text]

    # Simple decoding: ASCII code -> character
    def decode(ids, **kwargs):
        if isinstance(ids, int):
            ids = [ids]
        return "".join(chr(i + 32) for i in ids)

    tokenizer.encode = encode
    tokenizer.decode = decode

    return tokenizer


@pytest.fixture
def mock_model():
    """Mock transformer model for testing."""
    model = MagicMock()

    # Mock forward pass - return random logits
    def forward(*args, **kwargs):
        batch_size = 1
        vocab_size = 100
        logits = np.random.randn(batch_size, vocab_size)
        result = MagicMock()
        result.logits = MagicMock()
        result.logits.__getitem__ = lambda self, idx: logits
        return result

    model.return_value = MagicMock(logits=np.random.randn(1, 100))
    return model


# =============================================================================
# Probability Distribution Fixtures
# =============================================================================


@pytest.fixture
def uniform_distribution() -> np.ndarray:
    """Uniform probability distribution."""
    n = 100
    return np.ones(n) / n


@pytest.fixture
def peaked_distribution() -> np.ndarray:
    """Distribution with clear peak."""
    n = 100
    probs = np.ones(n) * 0.001
    probs[42] = 0.9  # Peak at index 42
    return probs / probs.sum()


@pytest.fixture
def random_distribution() -> np.ndarray:
    """Random probability distribution."""
    probs = np.random.dirichlet(np.ones(100))
    return probs


# =============================================================================
# Synthetic Trajectory Data
# =============================================================================


@pytest.fixture
def synthetic_trajectories() -> List[String]:
    """List of synthetic trajectory strings."""
    return [
        String(tokens=("The", " ", "cat", " ", "sat", ".")),
        String(tokens=("The", " ", "dog", " ", "ran", ".")),
        String(tokens=("A", " ", "bird", " ", "flew", ".")),
        String(tokens=("The", " ", "cat", " ", "ate", ".")),
        String(tokens=("A", " ", "dog", " ", "slept", ".")),
    ]


@pytest.fixture
def synthetic_probabilities() -> np.ndarray:
    """Probability distribution over synthetic trajectories."""
    probs = np.array([0.4, 0.25, 0.15, 0.12, 0.08])
    return probs / probs.sum()


@pytest.fixture
def synthetic_scores() -> np.ndarray:
    """Synthetic compliance scores for trajectories."""
    return np.array([0.8, 0.6, 0.3, 0.7, 0.4])


# =============================================================================
# Structure/System Fixtures
# =============================================================================


@pytest.fixture
def simple_compliance_fn():
    """Simple compliance function for testing."""

    def compliance(string: String) -> float:
        text = string.to_text().lower()
        # Simple heuristic: length-based compliance
        return min(1.0, len(text) / 20)

    return compliance


@pytest.fixture
def keyword_compliance_fn():
    """Keyword-based compliance function."""

    def compliance(string: String) -> float:
        text = string.to_text().lower()
        keywords = ["cat", "dog", "happy", "good"]
        count = sum(1 for k in keywords if k in text)
        return min(1.0, count / len(keywords))

    return compliance


# =============================================================================
# Test Configuration
# =============================================================================


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Directory for test data files."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture(scope="session")
def test_output_dir() -> Path:
    """Directory for test output files."""
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


# =============================================================================
# Numpy Test Utilities
# =============================================================================


@pytest.fixture
def assert_array_close():
    """Fixture for array comparison with tolerance."""

    def _assert_close(actual, expected, rtol=1e-5, atol=1e-8):
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)

    return _assert_close


@pytest.fixture
def assert_probability_distribution():
    """Fixture to assert valid probability distribution."""

    def _assert_prob_dist(probs, atol=1e-6):
        assert np.all(probs >= 0), "Probabilities must be non-negative"
        assert np.abs(np.sum(probs) - 1.0) < atol, "Probabilities must sum to 1"

    return _assert_prob_dist


# =============================================================================
# Markers
# =============================================================================


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "requires_gpu: marks tests that require GPU")
    config.addinivalue_line(
        "markers", "requires_model: marks tests that require downloading models"
    )
