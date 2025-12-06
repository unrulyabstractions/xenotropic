"""
Tests for similarity-based structures.

Tests for xenotechnics/structures/similarity.py
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from xenotechnics.common import String
from xenotechnics.structures.similarity import (
    MultiReferenceSimilarityStructure,
    SimilarityStructure,
    default_string_to_input,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __call__(
        self,
        text: str,
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt",
    ):
        return {
            "input_ids": torch.zeros(
                1, min(len(text) + 2, max_length), dtype=torch.long
            )
        }


class SimpleEmbedder(nn.Module):
    """Simple mock embedding model for testing."""

    def __init__(self, embedding_dim: int = 8, fixed_output: torch.Tensor = None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fixed_output = fixed_output
        # Use a parameter so device detection works
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size = x.shape[0]
        if self.fixed_output is not None:
            return self.fixed_output.unsqueeze(0).expand(batch_size, -1)
        # Return normalized random-ish embedding based on input shape
        embedding = torch.randn(batch_size, self.embedding_dim)
        return embedding / embedding.norm(dim=-1, keepdim=True)


class TestDefaultStringToInput:
    """Test default_string_to_input function."""

    def test_creates_callable(self):
        """Test that function returns a callable."""
        tokenizer = MockTokenizer()
        convert = default_string_to_input(tokenizer)

        assert callable(convert)

    def test_converts_string(self):
        """Test conversion of String to tensor."""
        tokenizer = MockTokenizer()
        convert = default_string_to_input(tokenizer)

        string = String(tokens=("hello", " ", "world"))
        result = convert(string)

        assert isinstance(result, torch.Tensor)


class TestSimilarityStructure:
    """Test SimilarityStructure class."""

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer()

    @pytest.fixture
    def reference_embedding(self):
        """Create a fixed reference embedding."""
        return torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_init_with_tokenizer(self, tokenizer, reference_embedding):
        """Test initialization with tokenizer."""
        model = SimpleEmbedder()
        structure = SimilarityStructure(
            model=model,
            reference_embedding=reference_embedding,
            tokenizer=tokenizer,
            name="test_similarity",
        )

        assert structure.name == "test_similarity"
        assert structure.similarity_metric == "cosine"

    def test_init_no_converter_raises(self, reference_embedding):
        """Test that initialization without tokenizer or string_to_input raises."""
        model = SimpleEmbedder()
        with pytest.raises(ValueError, match="Must provide either"):
            SimilarityStructure(model=model, reference_embedding=reference_embedding)

    def test_compliance_cosine_returns_float(self, tokenizer, reference_embedding):
        """Test cosine similarity compliance returns a float."""
        model = SimpleEmbedder()
        structure = SimilarityStructure(
            model=model,
            reference_embedding=reference_embedding,
            tokenizer=tokenizer,
            similarity_metric="cosine",
        )

        string = String(tokens=("test",))
        result = structure.compliance(string)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_compliance_identical_embedding(self, tokenizer):
        """Test compliance is high for identical embeddings."""
        fixed_embedding = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        model = SimpleEmbedder(fixed_output=fixed_embedding)

        structure = SimilarityStructure(
            model=model,
            reference_embedding=fixed_embedding.clone(),
            tokenizer=tokenizer,
            similarity_metric="cosine",
        )

        string = String(tokens=("test",))
        result = structure.compliance(string)

        # Identical embeddings should have cosine similarity of 1.0
        # Mapped to [0, 1] gives 1.0
        assert result == pytest.approx(1.0, rel=0.01)

    def test_compliance_opposite_embedding(self, tokenizer):
        """Test compliance is low for opposite embeddings."""
        ref_embedding = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Model outputs opposite direction
        model = SimpleEmbedder(
            fixed_output=torch.tensor([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )

        structure = SimilarityStructure(
            model=model,
            reference_embedding=ref_embedding,
            tokenizer=tokenizer,
            similarity_metric="cosine",
        )

        string = String(tokens=("test",))
        result = structure.compliance(string)

        # Opposite embeddings have cosine similarity -1.0
        # Mapped to [0, 1] gives 0.0
        assert result == pytest.approx(0.0, rel=0.01)

    def test_compliance_dot_metric(self, tokenizer):
        """Test dot product similarity metric."""
        ref_embedding = torch.tensor([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        model = SimpleEmbedder(fixed_output=ref_embedding)

        structure = SimilarityStructure(
            model=model,
            reference_embedding=ref_embedding.clone(),
            tokenizer=tokenizer,
            similarity_metric="dot",
        )

        string = String(tokens=("test",))
        result = structure.compliance(string)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_compliance_euclidean_metric(self, tokenizer):
        """Test euclidean distance metric."""
        ref_embedding = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        model = SimpleEmbedder(fixed_output=ref_embedding)

        structure = SimilarityStructure(
            model=model,
            reference_embedding=ref_embedding.clone(),
            tokenizer=tokenizer,
            similarity_metric="euclidean",
        )

        string = String(tokens=("test",))
        result = structure.compliance(string)

        # Identical embeddings have distance 0, similarity exp(0) = 1.0
        assert result == pytest.approx(1.0, rel=0.01)

    def test_unknown_metric_raises(self, tokenizer, reference_embedding):
        """Test unknown similarity metric raises."""
        model = SimpleEmbedder()
        structure = SimilarityStructure(
            model=model,
            reference_embedding=reference_embedding,
            tokenizer=tokenizer,
            similarity_metric="invalid",
        )

        string = String(tokens=("test",))
        with pytest.raises(ValueError, match="Unknown similarity metric"):
            structure.compliance(string)

    def test_from_reference_string(self, tokenizer):
        """Test from_reference_string class method."""
        model = SimpleEmbedder()
        ref_string = String(tokens=("reference", " ", "text"))

        structure = SimilarityStructure.from_reference_string(
            model=model,
            reference_string=ref_string,
            tokenizer=tokenizer,
            name="from_string",
        )

        assert structure.name == "from_string"
        assert structure.reference_embedding is not None

    def test_from_reference_string_no_converter_raises(self):
        """Test from_reference_string without tokenizer raises."""
        model = SimpleEmbedder()
        ref_string = String(tokens=("test",))

        with pytest.raises(ValueError, match="Must provide either"):
            SimilarityStructure.from_reference_string(
                model=model, reference_string=ref_string
            )


class TestMultiReferenceSimilarityStructure:
    """Test MultiReferenceSimilarityStructure class."""

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer()

    @pytest.fixture
    def reference_embeddings(self):
        """Create list of reference embeddings."""
        return [
            torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]

    def test_init(self, tokenizer, reference_embeddings):
        """Test initialization."""
        model = SimpleEmbedder()
        structure = MultiReferenceSimilarityStructure(
            model=model,
            reference_embeddings=reference_embeddings,
            tokenizer=tokenizer,
            aggregation="max",
        )

        assert len(structure.reference_embeddings) == 3
        assert structure.aggregation == "max"

    def test_init_no_converter_raises(self, reference_embeddings):
        """Test initialization without tokenizer raises."""
        model = SimpleEmbedder()
        with pytest.raises(ValueError, match="Must provide either"):
            MultiReferenceSimilarityStructure(
                model=model, reference_embeddings=reference_embeddings
            )

    def test_compliance_max_aggregation(self, tokenizer, reference_embeddings):
        """Test max aggregation."""
        # Model outputs embedding close to first reference
        model = SimpleEmbedder(
            fixed_output=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )

        structure = MultiReferenceSimilarityStructure(
            model=model,
            reference_embeddings=reference_embeddings,
            tokenizer=tokenizer,
            aggregation="max",
        )

        string = String(tokens=("test",))
        result = structure.compliance(string)

        # Should match first reference, so max similarity should be ~1.0
        assert result > 0.9

    def test_compliance_mean_aggregation(self, tokenizer, reference_embeddings):
        """Test mean aggregation."""
        model = SimpleEmbedder(
            fixed_output=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )

        structure = MultiReferenceSimilarityStructure(
            model=model,
            reference_embeddings=reference_embeddings,
            tokenizer=tokenizer,
            aggregation="mean",
        )

        string = String(tokens=("test",))
        result = structure.compliance(string)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_compliance_min_aggregation(self, tokenizer, reference_embeddings):
        """Test min aggregation."""
        model = SimpleEmbedder(
            fixed_output=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )

        structure = MultiReferenceSimilarityStructure(
            model=model,
            reference_embeddings=reference_embeddings,
            tokenizer=tokenizer,
            aggregation="min",
        )

        string = String(tokens=("test",))
        result = structure.compliance(string)

        assert isinstance(result, float)
        # Min should be lower than max (orthogonal to some references)
        assert result < 0.6

    def test_unknown_aggregation_raises(self, tokenizer, reference_embeddings):
        """Test unknown aggregation raises."""
        model = SimpleEmbedder()
        structure = MultiReferenceSimilarityStructure(
            model=model,
            reference_embeddings=reference_embeddings,
            tokenizer=tokenizer,
            aggregation="invalid",
        )

        string = String(tokens=("test",))
        with pytest.raises(ValueError, match="Unknown aggregation"):
            structure.compliance(string)

    def test_unknown_metric_raises(self, tokenizer, reference_embeddings):
        """Test unknown similarity metric raises."""
        model = SimpleEmbedder()
        structure = MultiReferenceSimilarityStructure(
            model=model,
            reference_embeddings=reference_embeddings,
            tokenizer=tokenizer,
            similarity_metric="invalid",
        )

        string = String(tokens=("test",))
        with pytest.raises(ValueError, match="Unknown similarity metric"):
            structure.compliance(string)
