"""
Tests for cloud generators.

Tests for exploration/generators/cloud.py
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class MockInferenceResponse:
    """Mock response from HuggingFace InferenceClient."""

    def __init__(self, generated_text="Hello", tokens=None, logprobs=None, n_tokens=3):
        self.generated_text = generated_text
        self.details = MagicMock()
        if tokens is None:
            # Create mock tokens with logprobs
            self.details.tokens = []
            for i in range(n_tokens):
                tok = MagicMock()
                tok.text = f"tok{i}"
                tok.logprob = logprobs[i] if logprobs else -0.5
                self.details.tokens.append(tok)
        else:
            self.details.tokens = tokens


class TestCloudScore:
    """Test CloudScore dataclass."""

    def test_creation(self):
        """Test CloudScore creation."""
        from exploration.generators.cloud import CloudScore

        score = CloudScore(text="hello", prob=0.5, logprob=-0.69, n_tokens=3)

        assert score.text == "hello"
        assert score.prob == 0.5
        assert score.logprob == -0.69
        assert score.n_tokens == 3

    def test_default_n_tokens(self):
        """Test CloudScore default n_tokens."""
        from exploration.generators.cloud import CloudScore

        score = CloudScore(text="hi", prob=0.8, logprob=-0.22)

        assert score.n_tokens == 1


class TestCloudGreedyGenerator:
    """Test CloudGreedyGenerator."""

    def test_init(self):
        """Test initialization."""
        from exploration.generators.cloud import CloudGreedyGenerator

        gen = CloudGreedyGenerator("test-model")

        assert gen.model_name == "test-model"
        assert gen._client is None  # Lazy loaded

    @patch("exploration.generators.cloud._get_inference_client")
    def test_client_lazy_loading(self, mock_get_client):
        """Test that client is lazy loaded."""
        from exploration.generators.cloud import CloudGreedyGenerator

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        gen = CloudGreedyGenerator("test-model")
        assert gen._client is None

        # Access client
        client = gen.client
        assert client == mock_client
        mock_get_client.assert_called_once()

        # Second access doesn't reload
        client2 = gen.client
        assert client2 == mock_client
        mock_get_client.assert_called_once()  # Still once

    @patch("exploration.generators.cloud._get_inference_client")
    def test_generate(self, mock_get_client):
        """Test generate method."""
        from exploration.generators.cloud import CloudGreedyGenerator

        mock_client = MagicMock()
        mock_client.text_generation.return_value = MockInferenceResponse(
            generated_text="World", n_tokens=2
        )
        mock_get_client.return_value = mock_client

        gen = CloudGreedyGenerator("test-model")
        result = gen.generate("Hello", max_new_tokens=10)

        assert result.text == "World"
        assert result.n_tokens == 2
        mock_client.text_generation.assert_called_once()

    @patch("exploration.generators.cloud._get_inference_client")
    def test_generate_error_handling(self, mock_get_client):
        """Test generate handles errors gracefully."""
        from exploration.generators.cloud import CloudGreedyGenerator

        mock_client = MagicMock()
        mock_client.text_generation.side_effect = Exception("API Error")
        mock_get_client.return_value = mock_client

        gen = CloudGreedyGenerator("test-model")
        result = gen.generate("Hello", max_new_tokens=10)

        assert result.text == ""
        assert result.prob == 0.0

    @patch("exploration.generators.cloud._get_inference_client")
    def test_greedy_next(self, mock_get_client):
        """Test greedy_next method."""
        from exploration.generators.cloud import CloudGreedyGenerator

        tok = MagicMock()
        tok.text = "X"
        tok.logprob = -0.5

        response = MagicMock()
        response.details = MagicMock()
        response.details.tokens = [tok]

        mock_client = MagicMock()
        mock_client.text_generation.return_value = response
        mock_get_client.return_value = mock_client

        gen = CloudGreedyGenerator("test-model")
        result = gen.greedy_next("Hello")

        assert result.text == "X"
        assert abs(result.logprob - (-0.5)) < 0.01


class TestCloudScorerGenerator:
    """Test CloudScorerGenerator."""

    def test_init(self):
        """Test initialization."""
        from exploration.generators.cloud import CloudScorerGenerator

        gen = CloudScorerGenerator("test-model")

        assert gen.model_name == "test-model"
        assert gen._client is None

    @patch("exploration.generators.cloud._get_inference_client")
    def test_score_returns_dict(self, mock_get_client):
        """Test score returns expected dict structure."""
        from exploration.generators.cloud import CloudScorerGenerator

        response = MockInferenceResponse(
            generated_text="continuation", n_tokens=3, logprobs=[-0.5, -0.3, -0.2]
        )

        mock_client = MagicMock()
        mock_client.text_generation.return_value = response
        mock_get_client.return_value = mock_client

        gen = CloudScorerGenerator("test-model")
        result = gen.score("prompt", "continuation")

        assert "logprob" in result
        assert "prob" in result
        assert "n_tokens" in result

    @patch("exploration.generators.cloud._get_inference_client")
    def test_score_error_handling(self, mock_get_client):
        """Test score handles errors gracefully."""
        from exploration.generators.cloud import CloudScorerGenerator

        mock_client = MagicMock()
        mock_client.text_generation.side_effect = Exception("API Error")
        mock_get_client.return_value = mock_client

        gen = CloudScorerGenerator("test-model")
        result = gen.score("prompt", "continuation")

        assert result["prob"] == 0.0
        assert "error" in result

    @patch("exploration.generators.cloud._get_inference_client")
    def test_score_continuations_classmethod(self, mock_get_client):
        """Test score_continuations class method."""
        from exploration.generators.cloud import CloudScorerGenerator

        response = MockInferenceResponse(generated_text="a", n_tokens=1)

        mock_client = MagicMock()
        mock_client.text_generation.return_value = response
        mock_get_client.return_value = mock_client

        result = CloudScorerGenerator.score_continuations(
            model_name="test-model",
            prompt="Hello",
            continuations=["a", "b"],
        )

        assert "prompt" in result
        assert "model" in result
        assert "scores" in result
        assert "ranking" in result
        assert len(result["scores"]) == 2
