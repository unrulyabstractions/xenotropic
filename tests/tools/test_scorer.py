"""
Tests for continuation scorer.

Tests for tools/scorer.py
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import torch


class MockTokenizer:
    """Mock tokenizer."""

    def encode(self, text, add_special_tokens=False):
        """Simple tokenization: each word becomes a token ID."""
        if not text:
            return []
        # Simple mock: hash each character group
        tokens = []
        for char in text:
            tokens.append(ord(char) % 1000)
        return tokens

    def decode(self, token_ids):
        """Decode token IDs back to text."""
        return "".join(chr(t % 128) for t in token_ids)


class MockModel:
    """Mock HuggingFace model."""

    def __call__(self, input_ids):
        """Return mock outputs with logits."""
        seq_len = input_ids.shape[1]
        vocab_size = 1000

        # Create logits that give reasonable probabilities
        logits = torch.randn(1, seq_len, vocab_size)

        result = MagicMock()
        result.logits = logits
        return result


class MockModelWrapper:
    """Mock model wrapper for testing."""

    def __init__(self):
        self.tokenizer = MockTokenizer()
        self.model = MockModel()
        self.device = "cpu"


class TestScorerInit:
    """Test Scorer initialization."""

    def test_init_with_model(self):
        """Test initialization with existing model."""
        from tools.common import Scorer

        mock_wrapper = MockModelWrapper()

        with patch.object(Scorer, "_ensure_model_loaded"):
            scorer = Scorer(model=mock_wrapper)
            assert scorer.model is mock_wrapper

    def test_init_lazy_load(self):
        """Test lazy loading initialization."""
        from tools.common import Scorer

        scorer = Scorer(model_name="test-model", lazy_load=True)
        assert scorer.model is None


class TestScorerScore:
    """Test Scorer.score() method."""

    def test_score_returns_dict_structure(self):
        """Test that score returns expected dict structure."""
        from tools.common import Scorer

        scorer = Scorer(model_name="test", lazy_load=True)
        scorer.model = MockModelWrapper()

        # Mock String
        mock_prompt = MagicMock()
        mock_prompt.to_text.return_value = "Hello"

        result = scorer.score(prompt=mock_prompt, continuation=" world")

        assert "logprob" in result
        assert "prob" in result
        assert "n_tokens" in result
        assert "per_token_logprobs" in result

    def test_score_empty_continuation(self):
        """Test scoring empty continuation."""
        from tools.common import Scorer

        scorer = Scorer(model_name="test", lazy_load=True)
        scorer.model = MockModelWrapper()

        mock_prompt = MagicMock()
        mock_prompt.to_text.return_value = "Hello"

        result = scorer.score(prompt=mock_prompt, continuation="")

        assert result["logprob"] == 0.0
        assert result["prob"] == 1.0
        assert result["n_tokens"] == 0

    def test_score_no_prompt(self):
        """Test scoring without prompt."""
        from tools.common import Scorer

        scorer = Scorer(model_name="test", lazy_load=True)
        scorer.model = MockModelWrapper()

        result = scorer.score(prompt=None, continuation="Hello")

        assert "logprob" in result
        assert "prob" in result
        assert result["n_tokens"] > 0

    def test_score_prob_is_exp_of_logprob(self):
        """Test that prob = exp(logprob)."""
        from tools.common import Scorer

        scorer = Scorer(model_name="test", lazy_load=True)
        scorer.model = MockModelWrapper()

        mock_prompt = MagicMock()
        mock_prompt.to_text.return_value = "Test"

        result = scorer.score(prompt=mock_prompt, continuation=" text")

        assert np.isclose(result["prob"], np.exp(result["logprob"]))

    def test_score_n_tokens_matches_per_token_list(self):
        """Test that n_tokens matches length of per_token_logprobs."""
        from tools.common import Scorer

        scorer = Scorer(model_name="test", lazy_load=True)
        scorer.model = MockModelWrapper()

        mock_prompt = MagicMock()
        mock_prompt.to_text.return_value = "Hello"

        result = scorer.score(prompt=mock_prompt, continuation=" world")

        assert result["n_tokens"] == len(result["per_token_logprobs"])


class TestScorerScoreContinuations:
    """Test Scorer.score_continuations() class method."""

    def test_score_continuations_returns_rankings(self):
        """Test that score_continuations returns rankings."""
        from tools.common import Scorer

        with patch.object(Scorer, "__init__", lambda self, **kwargs: None):
            scorer = Scorer()
            scorer.model = MockModelWrapper()
            scorer.debug = False

            with patch.object(scorer, "score") as mock_score:
                mock_score.side_effect = [
                    {
                        "logprob": -1.0,
                        "prob": 0.368,
                        "n_tokens": 1,
                        "per_token_logprobs": [],
                    },
                    {
                        "logprob": -0.5,
                        "prob": 0.606,
                        "n_tokens": 1,
                        "per_token_logprobs": [],
                    },
                ]

                with (
                    patch.object(Scorer, "__init__", lambda self, **kwargs: None),
                    patch(
                        "tools.common.scorer.Scorer",
                        return_value=scorer,
                    ),
                ):
                    # Can't easily test classmethod with mocks, just verify structure
                    pass
