"""
Tests for ModelRunner.

Tests for exploration/common/model_runner.py

Note: These tests mock transformer_lens at the module level to avoid
import issues when the library isn't installed.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

# Skip all tests in this module if we can't properly mock transformer_lens
# due to import caching issues
pytestmark = pytest.mark.skipif(
    "transformer_lens" not in sys.modules
    and "exploration.common.model_runner" in sys.modules,
    reason="Cannot properly mock transformer_lens after initial import",
)


def create_mock_model():
    """Create a mock model with default configuration."""
    mock_model = MagicMock()
    mock_model.cfg.n_layers = 12
    mock_model.cfg.d_model = 768
    mock_model.cfg.d_vocab = 50257
    return mock_model


@pytest.fixture(scope="module")
def mock_transformer_lens():
    """
    Mock transformer_lens module for the entire test module.

    Uses module scope to avoid reimporting issues with PyO3/Rust extensions.
    """
    mock_module = MagicMock()
    mock_hooked_transformer = MagicMock()
    mock_module.HookedTransformer = mock_hooked_transformer

    # Configure default mock model
    mock_model = create_mock_model()
    mock_hooked_transformer.from_pretrained.return_value = mock_model

    with patch.dict(sys.modules, {"transformer_lens": mock_module}):
        yield mock_hooked_transformer


@pytest.fixture
def fresh_mock(mock_transformer_lens):
    """Reset mock state for each test while reusing the module mock."""
    mock_transformer_lens.reset_mock()
    mock_model = create_mock_model()
    mock_transformer_lens.from_pretrained.return_value = mock_model
    return mock_transformer_lens


class TestModelRunnerInit:
    """Test ModelRunner initialization."""

    def test_init_auto_detects_device_cpu(self, fresh_mock):
        """Test device auto-detection falls back to CPU."""
        from exploration.common.model_runner import ModelRunner

        with patch("torch.backends.mps.is_available", return_value=False):
            with patch("torch.cuda.is_available", return_value=False):
                runner = ModelRunner(model_name="test-model")

        assert runner.device == "cpu"
        assert runner.dtype == torch.float32

    def test_init_with_explicit_device(self, fresh_mock):
        """Test initialization with explicit device."""
        from exploration.common.model_runner import ModelRunner

        runner = ModelRunner(model_name="test-model", device="cpu", dtype=torch.float32)

        assert runner.device == "cpu"
        assert runner.dtype == torch.float32
        assert runner.model_name == "test-model"

    def test_init_loads_model(self, fresh_mock):
        """Test that init loads the model."""
        from exploration.common.model_runner import ModelRunner

        runner = ModelRunner(model_name="test-model", device="cpu")

        fresh_mock.from_pretrained.assert_called_with(
            "test-model", device="cpu", dtype=torch.float32
        )
        runner.model.eval.assert_called()

    def test_init_auto_detects_mps(self, fresh_mock):
        """Test device auto-detection for MPS."""
        from exploration.common.model_runner import ModelRunner

        with patch("torch.backends.mps.is_available", return_value=True):
            runner = ModelRunner(model_name="test-model")

        assert runner.device == "mps"
        assert runner.dtype == torch.float16

    def test_init_auto_detects_cuda(self, fresh_mock):
        """Test device auto-detection for CUDA."""
        from exploration.common.model_runner import ModelRunner

        with patch("torch.backends.mps.is_available", return_value=False):
            with patch("torch.cuda.is_available", return_value=True):
                runner = ModelRunner(model_name="test-model")

        assert runner.device == "cuda"
        assert runner.dtype == torch.float16


class TestModelRunnerProperties:
    """Test ModelRunner properties."""

    def test_n_layers(self, fresh_mock):
        """Test n_layers property."""
        from exploration.common.model_runner import ModelRunner

        mock_model = fresh_mock.from_pretrained.return_value
        mock_model.cfg.n_layers = 24

        runner = ModelRunner(model_name="test-model", device="cpu")

        assert runner.n_layers == 24

    def test_d_model(self, fresh_mock):
        """Test d_model property."""
        from exploration.common.model_runner import ModelRunner

        mock_model = fresh_mock.from_pretrained.return_value
        mock_model.cfg.d_model = 1024

        runner = ModelRunner(model_name="test-model", device="cpu")

        assert runner.d_model == 1024

    def test_vocab_size(self, fresh_mock):
        """Test vocab_size property."""
        from exploration.common.model_runner import ModelRunner

        mock_model = fresh_mock.from_pretrained.return_value
        mock_model.cfg.d_vocab = 50257

        runner = ModelRunner(model_name="test-model", device="cpu")

        assert runner.vocab_size == 50257


class TestModelRunnerChatDetection:
    """Test chat model detection."""

    def test_detects_instruct_model(self, fresh_mock):
        """Test detection of instruction-tuned model."""
        from exploration.common.model_runner import ModelRunner

        runner = ModelRunner(model_name="Qwen/Qwen2.5-0.5B-Instruct", device="cpu")

        assert runner._is_chat_model is True

    def test_detects_base_model(self, fresh_mock):
        """Test detection of base model (not instruction-tuned)."""
        from exploration.common.model_runner import ModelRunner

        runner = ModelRunner(model_name="gpt2", device="cpu")

        assert runner._is_chat_model is False


class TestModelRunnerTokenize:
    """Test ModelRunner tokenization."""

    def test_tokenize(self, fresh_mock):
        """Test tokenization."""
        from exploration.common.model_runner import ModelRunner

        mock_model = fresh_mock.from_pretrained.return_value
        mock_model.to_tokens.return_value = torch.tensor([[1, 2, 3]])

        runner = ModelRunner(model_name="test-model", device="cpu")
        result = runner.tokenize("Hello world")

        mock_model.to_tokens.assert_called_with("Hello world", prepend_bos=True)
        assert result.shape == (1, 3)


class TestModelRunnerDecode:
    """Test ModelRunner decoding."""

    def test_decode(self, fresh_mock):
        """Test decoding."""
        from exploration.common.model_runner import ModelRunner

        mock_model = fresh_mock.from_pretrained.return_value
        mock_model.to_string.return_value = "Hello world"

        runner = ModelRunner(model_name="test-model", device="cpu")
        result = runner.decode(torch.tensor([1, 2, 3]))

        assert result == "Hello world"


class TestModelRunnerComputeDistribution:
    """Test ModelRunner.compute_distribution()."""

    def test_compute_distribution(self, fresh_mock):
        """Test distribution computation."""
        from exploration.common.model_runner import ModelRunner

        runner = ModelRunner(model_name="test-model", device="cpu")
        logits = torch.tensor([1.0, 2.0, 3.0])

        result = runner.compute_distribution(logits)

        # Should sum to 1
        assert result.sum().item() == pytest.approx(1.0)
        # Higher logit = higher probability
        assert result[2] > result[1] > result[0]


class TestModelRunnerGenerate:
    """Test ModelRunner.generate()."""

    def test_generate_basic(self, fresh_mock):
        """Test basic generation."""
        from exploration.common.model_runner import ModelRunner

        mock_model = fresh_mock.from_pretrained.return_value
        mock_model.to_tokens.return_value = torch.tensor([[1, 2, 3]])
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.to_string.return_value = "generated text"

        runner = ModelRunner(model_name="gpt2", device="cpu")
        result = runner.generate("Hello", max_new_tokens=10)

        assert result == "generated text"
        mock_model.generate.assert_called()


class TestModelRunnerGetNextTokenLogits:
    """Test ModelRunner.get_next_token_logits()."""

    def test_get_next_token_logits(self, fresh_mock):
        """Test getting next token logits."""
        from exploration.common.model_runner import ModelRunner

        mock_model = fresh_mock.from_pretrained.return_value
        mock_model.cfg.d_vocab = 100
        mock_model.return_value = torch.randn(1, 5, 100)

        runner = ModelRunner(model_name="test-model", device="cpu")
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])

        logits, kv_cache = runner.get_next_token_logits(input_ids)

        assert logits.shape == (1, 100)


class TestModelRunnerRunWithCache:
    """Test ModelRunner.run_with_cache()."""

    def test_run_with_cache_basic(self, fresh_mock):
        """Test running with cache."""
        from exploration.common.model_runner import ModelRunner

        # Create mock cache
        mock_cache = MagicMock()
        mock_cache.keys.return_value = ["blocks.0.hook_resid_post"]
        mock_cache.__getitem__ = lambda self, key: torch.randn(1, 5, 768)

        mock_model = fresh_mock.from_pretrained.return_value
        mock_model.to_tokens.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.run_with_cache.return_value = (torch.randn(1, 5, 50257), mock_cache)

        runner = ModelRunner(model_name="gpt2", device="cpu")
        logits, cache_dict = runner.run_with_cache("Hello world")

        assert logits.shape == (1, 5, 50257)
        assert "blocks.0.hook_resid_post" in cache_dict


class TestModelRunnerGetActivationNames:
    """Test ModelRunner.get_activation_names()."""

    def test_get_activation_names_default(self, fresh_mock):
        """Test getting activation names with defaults."""
        from exploration.common.model_runner import ModelRunner

        mock_model = fresh_mock.from_pretrained.return_value
        mock_model.cfg.n_layers = 4

        runner = ModelRunner(model_name="test-model", device="cpu")
        names = runner.get_activation_names()

        assert len(names) == 4  # All layers
        assert names[0] == "blocks.0.hook_resid_post"
        assert names[3] == "blocks.3.hook_resid_post"

    def test_get_activation_names_specific_layers(self, fresh_mock):
        """Test getting activation names for specific layers."""
        from exploration.common.model_runner import ModelRunner

        mock_model = fresh_mock.from_pretrained.return_value
        mock_model.cfg.n_layers = 12

        runner = ModelRunner(model_name="test-model", device="cpu")
        names = runner.get_activation_names(layers=[0, 6, 11])

        assert len(names) == 3
        assert names == [
            "blocks.0.hook_resid_post",
            "blocks.6.hook_resid_post",
            "blocks.11.hook_resid_post",
        ]

    def test_get_activation_names_multiple_components(self, fresh_mock):
        """Test getting activation names for multiple components."""
        from exploration.common.model_runner import ModelRunner

        mock_model = fresh_mock.from_pretrained.return_value
        mock_model.cfg.n_layers = 2

        runner = ModelRunner(model_name="test-model", device="cpu")
        names = runner.get_activation_names(components=["resid_post", "mlp_out"])

        assert len(names) == 4  # 2 layers * 2 components
        assert "blocks.0.hook_resid_post" in names
        assert "blocks.0.hook_mlp_out" in names
        assert "blocks.1.hook_resid_post" in names
        assert "blocks.1.hook_mlp_out" in names

    def test_get_activation_names_negative_index(self, fresh_mock):
        """Test getting activation names with negative layer index."""
        from exploration.common.model_runner import ModelRunner

        mock_model = fresh_mock.from_pretrained.return_value
        mock_model.cfg.n_layers = 12

        runner = ModelRunner(model_name="test-model", device="cpu")
        names = runner.get_activation_names(layers=[-1])

        assert names == ["blocks.11.hook_resid_post"]


class TestModelRunnerRepr:
    """Test ModelRunner.__repr__()."""

    def test_repr(self, fresh_mock):
        """Test __repr__ format."""
        from exploration.common.model_runner import ModelRunner

        runner = ModelRunner(model_name="test-model", device="cpu")
        rep = repr(runner)

        assert "ModelRunner" in rep
        assert "test-model" in rep
        assert "cpu" in rep
        assert "n_layers=12" in rep
        assert "d_model=768" in rep
