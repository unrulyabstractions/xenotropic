"""
Tests for model wrapper.

Tests for exploration/common/model.py
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch


class TestModelWrapperInit:
    """Test ModelWrapper initialization."""

    @patch("exploration.common.model.AutoModelForCausalLM")
    @patch("exploration.common.model.AutoTokenizer")
    def test_init_auto_detects_device_cpu(self, mock_tokenizer_cls, mock_model_cls):
        """Test device auto-detection falls back to CPU."""
        from exploration.common.model import ModelWrapper

        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()

        with patch("torch.backends.mps.is_available", return_value=False):
            with patch("torch.cuda.is_available", return_value=False):
                wrapper = ModelWrapper(model_name="test-model")

        assert wrapper.device == "cpu"
        assert wrapper.dtype == torch.float32

    @patch("exploration.common.model.AutoModelForCausalLM")
    @patch("exploration.common.model.AutoTokenizer")
    def test_init_with_explicit_device(self, mock_tokenizer_cls, mock_model_cls):
        """Test initialization with explicit device."""
        from exploration.common.model import ModelWrapper

        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()

        wrapper = ModelWrapper(
            model_name="test-model", device="cpu", dtype=torch.float32
        )

        assert wrapper.device == "cpu"
        assert wrapper.dtype == torch.float32
        assert wrapper.model_name == "test-model"

    @patch("exploration.common.model.AutoModelForCausalLM")
    @patch("exploration.common.model.AutoTokenizer")
    def test_init_loads_model_and_tokenizer(self, mock_tokenizer_cls, mock_model_cls):
        """Test that init loads both model and tokenizer."""
        from exploration.common.model import ModelWrapper

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = mock_model

        wrapper = ModelWrapper(model_name="test-model", device="cpu")

        mock_tokenizer_cls.from_pretrained.assert_called_once_with("test-model")
        mock_model_cls.from_pretrained.assert_called_once()
        mock_model.eval.assert_called_once()

    @patch("exploration.common.model.AutoModelForCausalLM")
    @patch("exploration.common.model.AutoTokenizer")
    def test_init_auto_detects_mps(self, mock_tokenizer_cls, mock_model_cls):
        """Test device auto-detection for MPS."""
        from exploration.common.model import ModelWrapper

        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()

        with patch("torch.backends.mps.is_available", return_value=True):
            wrapper = ModelWrapper(model_name="test-model")

        assert wrapper.device == "mps"
        assert wrapper.dtype == torch.float16

    @patch("exploration.common.model.AutoModelForCausalLM")
    @patch("exploration.common.model.AutoTokenizer")
    def test_init_auto_detects_cuda(self, mock_tokenizer_cls, mock_model_cls):
        """Test device auto-detection for CUDA."""
        from exploration.common.model import ModelWrapper

        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()

        with patch("torch.backends.mps.is_available", return_value=False):
            with patch("torch.cuda.is_available", return_value=True):
                wrapper = ModelWrapper(model_name="test-model")

        assert wrapper.device == "cuda"
        assert wrapper.dtype == torch.float16


class TestModelWrapperTokenizePrompt:
    """Test ModelWrapper.tokenize_prompt()."""

    @patch("exploration.common.model.AutoModelForCausalLM")
    @patch("exploration.common.model.AutoTokenizer")
    def test_tokenize_with_chat_template(self, mock_tokenizer_cls, mock_model_cls):
        """Test tokenization with chat template."""
        from exploration.common.model import ModelWrapper

        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = "template"
        mock_tokenizer.apply_chat_template.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = MagicMock()

        wrapper = ModelWrapper(model_name="test-model", device="cpu")
        result = wrapper.tokenize_prompt("Hello", use_chat_template=True)

        mock_tokenizer.apply_chat_template.assert_called_once()
        assert result.shape == (1, 3)

    @patch("exploration.common.model.AutoModelForCausalLM")
    @patch("exploration.common.model.AutoTokenizer")
    def test_tokenize_without_chat_template(self, mock_tokenizer_cls, mock_model_cls):
        """Test tokenization without chat template."""
        from exploration.common.model import ModelWrapper

        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None
        mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = MagicMock()

        wrapper = ModelWrapper(model_name="test-model", device="cpu")
        result = wrapper.tokenize_prompt("Hello", use_chat_template=True)

        mock_tokenizer.encode.assert_called_once()

    @patch("exploration.common.model.AutoModelForCausalLM")
    @patch("exploration.common.model.AutoTokenizer")
    def test_tokenize_disabled_chat_template(self, mock_tokenizer_cls, mock_model_cls):
        """Test tokenization with chat template disabled."""
        from exploration.common.model import ModelWrapper

        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = "template"
        mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = MagicMock()

        wrapper = ModelWrapper(model_name="test-model", device="cpu")
        wrapper.tokenize_prompt("Hello", use_chat_template=False)

        mock_tokenizer.encode.assert_called_once()


class TestModelWrapperGetNextTokenLogits:
    """Test ModelWrapper.get_next_token_logits()."""

    @patch("exploration.common.model.AutoModelForCausalLM")
    @patch("exploration.common.model.AutoTokenizer")
    def test_first_step_full_forward(self, mock_tokenizer_cls, mock_model_cls):
        """Test first step does full forward pass."""
        from exploration.common.model import ModelWrapper

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 5, 100)
        mock_output.past_key_values = ("kv_cache",)

        mock_model = MagicMock()
        mock_model.return_value = mock_output
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        wrapper = ModelWrapper(model_name="test-model", device="cpu")
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])

        logits, kv_cache = wrapper.get_next_token_logits(
            input_ids, past_key_values=None
        )

        # Should call model with full input
        mock_model.assert_called_once()
        call_kwargs = mock_model.call_args[1]
        assert "input_ids" in call_kwargs
        assert call_kwargs["use_cache"] is True

        assert logits.shape == (1, 100)
        assert kv_cache == ("kv_cache",)

    @patch("exploration.common.model.AutoModelForCausalLM")
    @patch("exploration.common.model.AutoTokenizer")
    def test_subsequent_step_uses_cache(self, mock_tokenizer_cls, mock_model_cls):
        """Test subsequent steps use KV cache."""
        from exploration.common.model import ModelWrapper

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 1, 100)
        mock_output.past_key_values = ("new_kv_cache",)

        mock_model = MagicMock()
        mock_model.return_value = mock_output
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        wrapper = ModelWrapper(model_name="test-model", device="cpu")
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        old_kv = ("old_kv_cache",)

        logits, kv_cache = wrapper.get_next_token_logits(
            input_ids, past_key_values=old_kv
        )

        # Should call model with only last token
        call_kwargs = mock_model.call_args[1]
        assert call_kwargs["input_ids"].shape == (1, 1)
        assert call_kwargs["past_key_values"] == old_kv


class TestModelWrapperDecodeTokens:
    """Test ModelWrapper.decode_tokens()."""

    @patch("exploration.common.model.AutoModelForCausalLM")
    @patch("exploration.common.model.AutoTokenizer")
    def test_decode_tokens(self, mock_tokenizer_cls, mock_model_cls):
        """Test token decoding."""
        from exploration.common.model import ModelWrapper

        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "Hello world"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = MagicMock()

        wrapper = ModelWrapper(model_name="test-model", device="cpu")
        result = wrapper.decode_tokens(torch.tensor([1, 2, 3]))

        assert result == "Hello world"
        mock_tokenizer.decode.assert_called_once()

    @patch("exploration.common.model.AutoModelForCausalLM")
    @patch("exploration.common.model.AutoTokenizer")
    def test_decode_skip_special_tokens(self, mock_tokenizer_cls, mock_model_cls):
        """Test decoding with skip_special_tokens."""
        from exploration.common.model import ModelWrapper

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = MagicMock()

        wrapper = ModelWrapper(model_name="test-model", device="cpu")
        wrapper.decode_tokens(torch.tensor([1, 2, 3]), skip_special_tokens=True)

        # Verify call was made with skip_special_tokens=True
        mock_tokenizer.decode.assert_called_once()
        call_args = mock_tokenizer.decode.call_args
        assert call_args[1]["skip_special_tokens"] is True


class TestModelWrapperComputeDistribution:
    """Test ModelWrapper.compute_distribution()."""

    @patch("exploration.common.model.AutoModelForCausalLM")
    @patch("exploration.common.model.AutoTokenizer")
    def test_compute_distribution(self, mock_tokenizer_cls, mock_model_cls):
        """Test distribution computation."""
        from exploration.common.model import ModelWrapper

        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()

        wrapper = ModelWrapper(model_name="test-model", device="cpu")
        logits = torch.tensor([1.0, 2.0, 3.0])

        result = wrapper.compute_distribution(logits)

        # Should sum to 1
        assert result.sum().item() == pytest.approx(1.0)
        # Higher logit = higher probability
        assert result[2] > result[1] > result[0]


class TestModelWrapperProperties:
    """Test ModelWrapper properties."""

    @patch("exploration.common.model.AutoModelForCausalLM")
    @patch("exploration.common.model.AutoTokenizer")
    def test_vocab_size(self, mock_tokenizer_cls, mock_model_cls):
        """Test vocab_size property."""
        from exploration.common.model import ModelWrapper

        mock_model = MagicMock()
        mock_model.config.vocab_size = 50000
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        wrapper = ModelWrapper(model_name="test-model", device="cpu")

        assert wrapper.vocab_size == 50000

    @patch("exploration.common.model.AutoModelForCausalLM")
    @patch("exploration.common.model.AutoTokenizer")
    def test_eos_token_id(self, mock_tokenizer_cls, mock_model_cls):
        """Test eos_token_id property."""
        from exploration.common.model import ModelWrapper

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = MagicMock()

        wrapper = ModelWrapper(model_name="test-model", device="cpu")

        assert wrapper.eos_token_id == 2


# Check if torch version supports model loading
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
TORCH_TOO_OLD = TORCH_VERSION < (2, 6)
SKIP_REASON = "Requires torch>=2.6 for model loading (CVE-2025-32434)"


@pytest.mark.slow
@pytest.mark.skipif(TORCH_TOO_OLD, reason=SKIP_REASON)
class TestModelWrapperIntegration:
    """Integration tests using tiny model."""

    @pytest.fixture
    def tiny_model(self):
        """Load tiny GPT2 model for testing."""
        from exploration.common.model import ModelWrapper

        return ModelWrapper(
            model_name="sshleifer/tiny-gpt2",
            device="cpu",
            dtype=torch.float32,
        )

    def test_real_tokenization(self, tiny_model):
        """Test tokenization with real model."""
        input_ids = tiny_model.tokenize_prompt("Hello world", use_chat_template=False)
        assert input_ids.shape[0] == 1
        assert input_ids.shape[1] > 0

    def test_real_inference(self, tiny_model):
        """Test inference with real model."""
        input_ids = tiny_model.tokenize_prompt("Hello", use_chat_template=False)
        logits, kv_cache = tiny_model.get_next_token_logits(input_ids)

        assert logits.shape[0] == 1
        assert logits.shape[1] == tiny_model.vocab_size
        assert kv_cache is not None

    def test_real_distribution(self, tiny_model):
        """Test distribution computation with real model."""
        input_ids = tiny_model.tokenize_prompt("Hello", use_chat_template=False)
        logits, _ = tiny_model.get_next_token_logits(input_ids)

        dist = tiny_model.compute_distribution(logits[0])

        assert dist.sum().item() == pytest.approx(1.0, abs=1e-5)
        assert (dist >= 0).all()

    def test_real_decode(self, tiny_model):
        """Test decoding with real model."""
        input_ids = tiny_model.tokenize_prompt("Hello", use_chat_template=False)
        decoded = tiny_model.decode_tokens(input_ids[0])

        assert isinstance(decoded, str)
        assert len(decoded) > 0
