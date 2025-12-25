"""
Tests for sampling generator.

Tests for exploration/generators/sampling.py
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import torch


class MockTokenizer:
    """Mock tokenizer."""

    def __init__(self):
        self.eos_token_id = 99

    def decode(self, token_ids):
        """Return string representation."""
        return f"tok_{token_ids[0]}"

    def __call__(self, text, return_tensors="pt"):
        """Tokenize text."""
        result = MagicMock()
        result.input_ids = torch.tensor([[1, 2, 3]])
        return result


class MockModelWrapper:
    """Mock model wrapper for testing."""

    def __init__(self):
        self.tokenizer = MockTokenizer()

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    def compute_distribution(self, logits):
        """Compute distribution from logits."""
        return torch.softmax(logits, dim=-1)


class TestSamplingGeneratorInitStrategyState:
    """Test SamplingGenerator._init_strategy_state()."""

    def test_default_parameters(self):
        """Test default parameter initialization."""
        from exploration.generators.sampling import SamplingGenerator

        with patch.object(SamplingGenerator, "__init__", lambda self, **kwargs: None):
            gen = SamplingGenerator()

            gen._init_strategy_state()

            assert gen.temperature == 1.0
            assert gen.top_k is None
            assert gen.top_p is None
            assert gen.seed is None

    def test_custom_parameters(self):
        """Test custom parameter initialization."""
        from exploration.generators.sampling import SamplingGenerator

        with patch.object(SamplingGenerator, "__init__", lambda self, **kwargs: None):
            gen = SamplingGenerator()

            gen._init_strategy_state(
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                seed=42,
            )

            assert gen.temperature == 0.7
            assert gen.top_k == 50
            assert gen.top_p == 0.9
            assert gen.seed == 42

    def test_seed_sets_random_state(self):
        """Test that seed sets random state."""
        from exploration.generators.sampling import SamplingGenerator

        with patch.object(SamplingGenerator, "__init__", lambda self, **kwargs: None):
            gen = SamplingGenerator()
            gen._init_strategy_state(seed=42)

            # Should be deterministic
            val1 = np.random.rand()
            gen._init_strategy_state(seed=42)
            val2 = np.random.rand()

            assert val1 == val2


class TestSamplingGeneratorStepImpl:
    """Test SamplingGenerator.step_impl()."""

    def test_step_samples_from_distribution(self):
        """Test that step samples from distribution."""
        from exploration.generators.sampling import SamplingGenerator

        with patch.object(SamplingGenerator, "__init__", lambda self, **kwargs: None):
            gen = SamplingGenerator()
            gen.model = MockModelWrapper()
            gen.current_node = MagicMock()
            gen.current_node.add_child.return_value = MagicMock()
            gen.distributions = []
            gen.step_count = 0
            gen.max_steps = 10
            gen.debug = False
            gen.temperature = 1.0
            gen.top_k = None
            gen.top_p = None

            # Create logits with clear preference
            logits = torch.zeros(1, 100)
            logits[0, 42] = 10.0

            np.random.seed(42)
            generated_ids = torch.tensor([[1, 2, 3]])
            result = gen.step_impl(logits, gen.model, generated_ids, verbose=False)

            assert result is not None
            assert 0 <= result[0].item() < 100

    def test_step_with_temperature(self):
        """Test step with temperature scaling."""
        from exploration.generators.sampling import SamplingGenerator

        with patch.object(SamplingGenerator, "__init__", lambda self, **kwargs: None):
            gen = SamplingGenerator()
            gen.model = MockModelWrapper()
            gen.current_node = MagicMock()
            gen.current_node.add_child.return_value = MagicMock()
            gen.distributions = []
            gen.step_count = 0
            gen.max_steps = 10
            gen.debug = False
            gen.temperature = 0.5  # Lower temperature = sharper distribution
            gen.top_k = None
            gen.top_p = None

            logits = torch.zeros(1, 100)
            logits[0, 42] = 5.0

            np.random.seed(42)
            generated_ids = torch.tensor([[1, 2, 3]])
            result = gen.step_impl(logits, gen.model, generated_ids, verbose=False)

            assert result is not None

    def test_step_with_top_k(self):
        """Test step with top-k filtering."""
        from exploration.generators.sampling import SamplingGenerator

        with patch.object(SamplingGenerator, "__init__", lambda self, **kwargs: None):
            gen = SamplingGenerator()
            gen.model = MockModelWrapper()
            gen.current_node = MagicMock()
            gen.current_node.add_child.return_value = MagicMock()
            gen.distributions = []
            gen.step_count = 0
            gen.max_steps = 10
            gen.debug = False
            gen.temperature = 1.0
            gen.top_k = 10  # Only consider top 10 tokens
            gen.top_p = None

            logits = torch.randn(1, 100)

            np.random.seed(42)
            generated_ids = torch.tensor([[1, 2, 3]])
            result = gen.step_impl(logits, gen.model, generated_ids, verbose=False)

            # Result should be one of top-k tokens
            assert result is not None

    def test_step_with_top_p(self):
        """Test step with top-p (nucleus) filtering."""
        from exploration.generators.sampling import SamplingGenerator

        with patch.object(SamplingGenerator, "__init__", lambda self, **kwargs: None):
            gen = SamplingGenerator()
            gen.model = MockModelWrapper()
            gen.current_node = MagicMock()
            gen.current_node.add_child.return_value = MagicMock()
            gen.distributions = []
            gen.step_count = 0
            gen.max_steps = 10
            gen.debug = False
            gen.temperature = 1.0
            gen.top_k = None
            gen.top_p = 0.9  # Nucleus sampling

            logits = torch.randn(1, 100)

            np.random.seed(42)
            generated_ids = torch.tensor([[1, 2, 3]])
            result = gen.step_impl(logits, gen.model, generated_ids, verbose=False)

            assert result is not None

    def test_step_stops_on_eos(self):
        """Test that step stops on EOS token."""
        from exploration.generators.sampling import SamplingGenerator

        with patch.object(SamplingGenerator, "__init__", lambda self, **kwargs: None):
            gen = SamplingGenerator()
            gen.model = MockModelWrapper()
            gen.current_node = MagicMock()
            gen.current_node.add_child.return_value = MagicMock()
            gen.distributions = []
            gen.step_count = 0
            gen.max_steps = 10
            gen.debug = False
            gen.temperature = 1.0
            gen.top_k = None
            gen.top_p = None

            # Make EOS token (99) very likely
            logits = torch.zeros(1, 100)
            logits[0, 99] = 100.0

            generated_ids = torch.tensor([[1, 2, 3]])
            result = gen.step_impl(logits, gen.model, generated_ids, verbose=False)

            assert result is None

    def test_step_stops_on_max_steps(self):
        """Test that step stops at max_steps."""
        from exploration.generators.sampling import SamplingGenerator

        with patch.object(SamplingGenerator, "__init__", lambda self, **kwargs: None):
            gen = SamplingGenerator()
            gen.model = MockModelWrapper()
            gen.current_node = MagicMock()
            gen.current_node.add_child.return_value = MagicMock()
            gen.distributions = []
            gen.step_count = 10
            gen.max_steps = 10
            gen.debug = False
            gen.temperature = 1.0
            gen.top_k = None
            gen.top_p = None

            logits = torch.zeros(1, 100)
            logits[0, 42] = 10.0

            generated_ids = torch.tensor([[1, 2, 3]])
            result = gen.step_impl(logits, gen.model, generated_ids, verbose=False)

            assert result is None

    def test_step_verbose_output(self, capsys):
        """Test verbose output during step."""
        from exploration.generators.sampling import SamplingGenerator

        with patch.object(SamplingGenerator, "__init__", lambda self, **kwargs: None):
            gen = SamplingGenerator()
            gen.model = MockModelWrapper()
            gen.current_node = MagicMock()
            gen.current_node.add_child.return_value = MagicMock()
            gen.distributions = []
            gen.step_count = 0
            gen.max_steps = 10
            gen.debug = False
            gen.temperature = 1.0
            gen.top_k = None
            gen.top_p = None

            logits = torch.zeros(1, 100)
            logits[0, 42] = 10.0

            generated_ids = torch.tensor([[1, 2, 3]])
            result = gen.step_impl(logits, gen.model, generated_ids, verbose=True)

            captured = capsys.readouterr()
            assert "Step" in captured.out
            assert "T=" in captured.out


class TestSamplingGeneratorTopFiltering:
    """Test top-k and top-p filtering edge cases."""

    def test_top_k_with_zero(self):
        """Test top-k with k=0 (disabled)."""
        from exploration.generators.sampling import SamplingGenerator

        with patch.object(SamplingGenerator, "__init__", lambda self, **kwargs: None):
            gen = SamplingGenerator()
            gen.model = MockModelWrapper()
            gen.current_node = MagicMock()
            gen.current_node.add_child.return_value = MagicMock()
            gen.distributions = []
            gen.step_count = 0
            gen.max_steps = 10
            gen.debug = False
            gen.temperature = 1.0
            gen.top_k = 0  # Disabled
            gen.top_p = None

            logits = torch.randn(1, 100)

            np.random.seed(42)
            generated_ids = torch.tensor([[1, 2, 3]])
            result = gen.step_impl(logits, gen.model, generated_ids, verbose=False)

            assert result is not None

    def test_top_p_at_boundary(self):
        """Test top-p at boundary values."""
        from exploration.generators.sampling import SamplingGenerator

        with patch.object(SamplingGenerator, "__init__", lambda self, **kwargs: None):
            gen = SamplingGenerator()
            gen.model = MockModelWrapper()
            gen.current_node = MagicMock()
            gen.current_node.add_child.return_value = MagicMock()
            gen.distributions = []
            gen.step_count = 0
            gen.max_steps = 10
            gen.debug = False
            gen.temperature = 1.0
            gen.top_k = None
            gen.top_p = 1.0  # Should include all tokens

            logits = torch.randn(1, 100)

            np.random.seed(42)
            generated_ids = torch.tensor([[1, 2, 3]])
            result = gen.step_impl(logits, gen.model, generated_ids, verbose=False)

            assert result is not None

    def test_combined_top_k_and_top_p(self):
        """Test combined top-k and top-p filtering."""
        from exploration.generators.sampling import SamplingGenerator

        with patch.object(SamplingGenerator, "__init__", lambda self, **kwargs: None):
            gen = SamplingGenerator()
            gen.model = MockModelWrapper()
            gen.current_node = MagicMock()
            gen.current_node.add_child.return_value = MagicMock()
            gen.distributions = []
            gen.step_count = 0
            gen.max_steps = 10
            gen.debug = False
            gen.temperature = 1.0
            gen.top_k = 50
            gen.top_p = 0.9

            logits = torch.randn(1, 100)

            np.random.seed(42)
            generated_ids = torch.tensor([[1, 2, 3]])
            result = gen.step_impl(logits, gen.model, generated_ids, verbose=False)

            assert result is not None
