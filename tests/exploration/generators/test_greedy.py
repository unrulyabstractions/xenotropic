"""
Tests for greedy generator.

Tests for exploration/generators/greedy.py
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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
        self.step_count = 0
        self.max_steps = 5

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    def tokenize_prompt(self, text, use_chat_template=False):
        """Tokenize prompt."""
        return torch.tensor([[1, 2, 3]])

    def compute_distribution(self, logits):
        """Compute distribution from logits."""
        return torch.softmax(logits, dim=-1)

    def decode_tokens(self, ids, skip_special_tokens=False):
        """Decode tokens to text."""
        return "Generated text"

    def get_next_token_logits(self, input_ids, past_key_values):
        """Return mock logits."""
        logits = torch.randn(1, 100)
        # Make token 42 most likely
        logits[0, 42] = 10.0
        return logits, None


class TestGreedyGeneratorStepImpl:
    """Test GreedyGenerator.step_impl()."""

    def test_step_selects_argmax(self):
        """Test that step selects argmax token."""
        from exploration.generators.greedy import GreedyGenerator

        # Create generator with mocked model
        with patch.object(GreedyGenerator, "__init__", lambda self, **kwargs: None):
            gen = GreedyGenerator()
            gen.model = MockModelWrapper()
            gen.current_node = MagicMock()
            gen.current_node.add_child.return_value = MagicMock()
            gen.distributions = []
            gen.step_count = 0
            gen.max_steps = 10
            gen.debug = False

            # Create logits with clear argmax
            logits = torch.zeros(1, 100)
            logits[0, 42] = 10.0

            generated_ids = torch.tensor([[1, 2, 3]])
            result = gen.step_impl(logits, gen.model, generated_ids, verbose=False)

            # Should return the argmax token
            assert result is not None
            assert result[0].item() == 42

    def test_step_stops_on_eos(self):
        """Test that step stops on EOS token."""
        from exploration.generators.greedy import GreedyGenerator

        with patch.object(GreedyGenerator, "__init__", lambda self, **kwargs: None):
            gen = GreedyGenerator()
            gen.model = MockModelWrapper()
            gen.current_node = MagicMock()
            gen.current_node.add_child.return_value = MagicMock()
            gen.distributions = []
            gen.step_count = 0
            gen.max_steps = 10
            gen.debug = False

            # Create logits where EOS (99) is most likely
            logits = torch.zeros(1, 100)
            logits[0, 99] = 10.0

            generated_ids = torch.tensor([[1, 2, 3]])
            result = gen.step_impl(logits, gen.model, generated_ids, verbose=False)

            # Should return None to stop
            assert result is None

    def test_step_stops_on_max_steps(self):
        """Test that step stops at max_steps."""
        from exploration.generators.greedy import GreedyGenerator

        with patch.object(GreedyGenerator, "__init__", lambda self, **kwargs: None):
            gen = GreedyGenerator()
            gen.model = MockModelWrapper()
            gen.current_node = MagicMock()
            gen.current_node.add_child.return_value = MagicMock()
            gen.distributions = []
            gen.step_count = 10
            gen.max_steps = 10
            gen.debug = False

            logits = torch.zeros(1, 100)
            logits[0, 42] = 10.0

            generated_ids = torch.tensor([[1, 2, 3]])
            result = gen.step_impl(logits, gen.model, generated_ids, verbose=False)

            # Should return None because step_count >= max_steps
            assert result is None

    def test_step_stops_on_eos_string(self):
        """Test that step stops on EOS string tokens."""
        from exploration.generators.greedy import GreedyGenerator

        with patch.object(GreedyGenerator, "__init__", lambda self, **kwargs: None):
            gen = GreedyGenerator()
            gen.model = MockModelWrapper()
            gen.model.tokenizer.eos_token_id = None  # No EOS token ID
            gen.model.tokenizer.decode = lambda ids: "<|im_end|>"  # Return EOS string
            gen.current_node = MagicMock()
            gen.current_node.add_child.return_value = MagicMock()
            gen.distributions = []
            gen.step_count = 0
            gen.max_steps = 10
            gen.debug = False

            logits = torch.zeros(1, 100)
            logits[0, 42] = 10.0

            generated_ids = torch.tensor([[1, 2, 3]])
            result = gen.step_impl(logits, gen.model, generated_ids, verbose=False)

            assert result is None

    def test_step_verbose_output(self, capsys):
        """Test verbose output during step."""
        from exploration.generators.greedy import GreedyGenerator

        with patch.object(GreedyGenerator, "__init__", lambda self, **kwargs: None):
            gen = GreedyGenerator()
            gen.model = MockModelWrapper()
            gen.current_node = MagicMock()
            gen.current_node.add_child.return_value = MagicMock()
            gen.distributions = []
            gen.step_count = 0
            gen.max_steps = 10
            gen.debug = False

            logits = torch.zeros(1, 100)
            logits[0, 42] = 10.0

            generated_ids = torch.tensor([[1, 2, 3]])
            gen.step_impl(logits, gen.model, generated_ids, verbose=True)

            captured = capsys.readouterr()
            assert "Step" in captured.out
