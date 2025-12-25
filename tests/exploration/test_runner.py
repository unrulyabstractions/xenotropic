"""
Tests for generation runner.

Tests for exploration/common/runner.py
"""

from __future__ import annotations

import torch

from exploration.common.runner import Runner


class MockModelWrapper:
    """Mock model wrapper for testing."""

    def __init__(self, eos_token_id=None):
        self._eos_token_id = eos_token_id
        self.call_count = 0

    @property
    def eos_token_id(self):
        return self._eos_token_id

    def get_next_token_logits(self, input_ids, past_key_values):
        """Return mock logits and updated KV cache."""
        self.call_count += 1
        logits = torch.randn(1, 100)
        new_kv = (torch.randn(1, 1, 10, 10),)
        return logits, new_kv


class TestRunner:
    """Test Runner class."""

    def test_init(self):
        """Test runner initialization."""
        model = MockModelWrapper()
        runner = Runner(model)
        assert runner.model is model

    def test_run_generation_stops_on_none(self):
        """Test that generation stops when step function returns None."""
        model = MockModelWrapper()
        runner = Runner(model)

        # Step function that stops after 3 iterations
        call_count = [0]

        def step_fn(logits, model, generated_ids):
            call_count[0] += 1
            if call_count[0] >= 3:
                return None
            return torch.tensor([[42]])

        input_ids = torch.tensor([[1, 2, 3]])
        result = runner.run_generation(input_ids, step_fn)

        # Should have called step 3 times (stops when it returns None)
        assert call_count[0] == 3
        # Result should have original 3 + 2 new tokens
        assert result.shape[1] == 5

    def test_run_generation_stops_on_eos(self):
        """Test that generation stops on EOS token."""
        eos_id = 99
        model = MockModelWrapper(eos_token_id=eos_id)
        runner = Runner(model)

        call_count = [0]

        def step_fn(logits, model, generated_ids):
            call_count[0] += 1
            if call_count[0] == 2:
                return torch.tensor([[eos_id]])  # Return EOS
            return torch.tensor([[42]])

        input_ids = torch.tensor([[1, 2, 3]])
        result = runner.run_generation(input_ids, step_fn)

        # Should have stopped after EOS
        assert call_count[0] == 2
        # Result includes original + 2 new tokens (including EOS)
        assert result.shape[1] == 5
        assert result[0, -1].item() == eos_id

    def test_run_generation_appends_tokens(self):
        """Test that tokens are appended correctly."""
        model = MockModelWrapper()
        runner = Runner(model)

        tokens = [10, 20, 30]
        idx = [0]

        def step_fn(logits, model, generated_ids):
            if idx[0] >= len(tokens):
                return None
            token = tokens[idx[0]]
            idx[0] += 1
            return torch.tensor([[token]])

        input_ids = torch.tensor([[1, 2]])
        result = runner.run_generation(input_ids, step_fn)

        # Should have original 2 + 3 new tokens
        assert result.shape[1] == 5
        assert result[0, 2].item() == 10
        assert result[0, 3].item() == 20
        assert result[0, 4].item() == 30

    def test_run_generation_no_eos_token(self):
        """Test generation without EOS token defined."""
        model = MockModelWrapper(eos_token_id=None)
        runner = Runner(model)

        call_count = [0]

        def step_fn(logits, model, generated_ids):
            call_count[0] += 1
            if call_count[0] >= 5:
                return None
            return torch.tensor([[call_count[0]]])

        input_ids = torch.tensor([[0]])
        result = runner.run_generation(input_ids, step_fn)

        # Should continue until step_fn returns None
        assert call_count[0] == 5

    def test_run_generation_preserves_device(self):
        """Test that output is on same device as input."""
        model = MockModelWrapper()
        runner = Runner(model)

        def step_fn(logits, model, generated_ids):
            return None

        input_ids = torch.tensor([[1, 2, 3]])
        result = runner.run_generation(input_ids, step_fn)

        assert result.device == input_ids.device

    def test_run_generation_immediate_stop(self):
        """Test immediate stop (step returns None on first call)."""
        model = MockModelWrapper()
        runner = Runner(model)

        def step_fn(logits, model, generated_ids):
            return None

        input_ids = torch.tensor([[1, 2, 3]])
        result = runner.run_generation(input_ids, step_fn)

        # Should just return original input
        assert result.shape[1] == 3
        assert torch.equal(result, input_ids)
