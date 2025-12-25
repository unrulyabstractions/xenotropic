"""
Tests for classifier-based structures.

Tests for xenotechnics/structures/classifier.py
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from xenotechnics.common import String
from xenotechnics.structures.classifier import (
    ClassifierStructure,
    MultiClassifierStructure,
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
        # Return mock encoded input
        return {
            "input_ids": torch.zeros(
                1, min(len(text) + 2, max_length), dtype=torch.long
            )
        }


class SimpleClassifier(nn.Module):
    """Simple mock classifier for testing."""

    def __init__(self, num_classes: int = 2, output_class: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.output_class = output_class
        # Use a parameter so device detection works
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Return logits where output_class has highest value
        batch_size = x.shape[0]
        logits = torch.zeros(batch_size, self.num_classes)
        logits[:, self.output_class] = 2.0  # High logit for target class
        return logits


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

    def test_respects_max_length(self):
        """Test max_length parameter."""
        tokenizer = MockTokenizer()
        convert = default_string_to_input(tokenizer, max_length=32)

        string = String(tokens=("test",))
        result = convert(string)

        assert result.shape[1] <= 32

    def test_device_parameter(self):
        """Test device parameter."""
        tokenizer = MockTokenizer()
        convert = default_string_to_input(tokenizer, device="cpu")

        string = String(tokens=("test",))
        result = convert(string)

        assert result.device.type == "cpu"


class TestClassifierStructure:
    """Test ClassifierStructure class."""

    @pytest.fixture
    def model(self):
        """Create mock classifier."""
        return SimpleClassifier(num_classes=3, output_class=1)

    @pytest.fixture
    def tokenizer(self):
        """Create mock tokenizer."""
        return MockTokenizer()

    def test_init_with_tokenizer(self, model, tokenizer):
        """Test initialization with tokenizer."""
        structure = ClassifierStructure(
            model=model,
            target_class=1,
            tokenizer=tokenizer,
            name="test_classifier",
        )

        assert structure.target_class == 1
        assert structure.name == "test_classifier"

    def test_init_with_string_to_input(self, model):
        """Test initialization with custom string_to_input."""

        def custom_convert(s: String) -> torch.Tensor:
            return torch.zeros(1, 10, dtype=torch.long)

        structure = ClassifierStructure(
            model=model, target_class=1, string_to_input=custom_convert
        )

        assert structure.string_to_input == custom_convert

    def test_init_no_converter_raises(self, model):
        """Test that initialization without tokenizer or string_to_input raises."""
        with pytest.raises(ValueError, match="Must provide either"):
            ClassifierStructure(model=model, target_class=1)

    def test_compliance_returns_float(self, model, tokenizer):
        """Test compliance returns a float."""
        structure = ClassifierStructure(
            model=model, target_class=1, tokenizer=tokenizer
        )

        string = String(tokens=("test",))
        result = structure.compliance(string)

        assert isinstance(result, float)

    def test_compliance_in_range(self, model, tokenizer):
        """Test compliance is in [0, 1]."""
        structure = ClassifierStructure(
            model=model, target_class=1, tokenizer=tokenizer
        )

        test_strings = [
            String(tokens=("hello",)),
            String(tokens=("world",)),
            String(tokens=("test", " ", "string")),
            String.empty(),
        ]

        for s in test_strings:
            result = structure.compliance(s)
            assert 0.0 <= result <= 1.0

    def test_compliance_high_for_target_class(self, tokenizer):
        """Test that compliance is high when model predicts target class."""
        # Model outputs high logit for class 1
        model = SimpleClassifier(num_classes=2, output_class=1)
        structure = ClassifierStructure(
            model=model, target_class=1, tokenizer=tokenizer
        )

        string = String(tokens=("test",))
        result = structure.compliance(string)

        # Should be > 0.5 since model outputs high for class 1
        assert result > 0.5

    def test_compliance_low_for_other_class(self, tokenizer):
        """Test that compliance is low for non-target class."""
        # Model outputs high logit for class 1
        model = SimpleClassifier(num_classes=2, output_class=1)
        structure = ClassifierStructure(
            model=model,
            target_class=0,
            tokenizer=tokenizer,  # Target is class 0
        )

        string = String(tokens=("test",))
        result = structure.compliance(string)

        # Should be < 0.5 since model outputs low for class 0
        assert result < 0.5


class TestMultiClassifierStructure:
    """Test MultiClassifierStructure class."""

    @pytest.fixture
    def models(self):
        """Create list of mock classifiers."""
        return [
            SimpleClassifier(num_classes=2, output_class=1),
            SimpleClassifier(num_classes=2, output_class=0),
        ]

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer()

    def test_init_with_tokenizer(self, models, tokenizer):
        """Test initialization."""
        structure = MultiClassifierStructure(
            models=models,
            target_classes=[1, 0],
            tokenizer=tokenizer,
            aggregation="mean",
        )

        assert len(structure.models) == 2
        assert structure.aggregation == "mean"

    def test_init_mismatched_lengths_raises(self, models, tokenizer):
        """Test that mismatched model/target_class lengths raise."""
        with pytest.raises(ValueError, match="Number of models must match"):
            MultiClassifierStructure(
                models=models,
                target_classes=[1],  # Wrong length
                tokenizer=tokenizer,
            )

    def test_compliance_mean_aggregation(self, tokenizer):
        """Test mean aggregation."""
        models = [
            SimpleClassifier(num_classes=2, output_class=1),
            SimpleClassifier(num_classes=2, output_class=1),
        ]

        structure = MultiClassifierStructure(
            models=models,
            target_classes=[1, 1],
            tokenizer=tokenizer,
            aggregation="mean",
        )

        string = String(tokens=("test",))
        result = structure.compliance(string)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_compliance_max_aggregation(self, tokenizer):
        """Test max aggregation."""
        models = [
            SimpleClassifier(num_classes=2, output_class=1),
            SimpleClassifier(num_classes=2, output_class=0),
        ]

        structure = MultiClassifierStructure(
            models=models, target_classes=[1, 0], tokenizer=tokenizer, aggregation="max"
        )

        string = String(tokens=("test",))
        result = structure.compliance(string)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_compliance_min_aggregation(self, tokenizer):
        """Test min aggregation."""
        models = [
            SimpleClassifier(num_classes=2, output_class=1),
            SimpleClassifier(num_classes=2, output_class=0),
        ]

        structure = MultiClassifierStructure(
            models=models, target_classes=[1, 0], tokenizer=tokenizer, aggregation="min"
        )

        string = String(tokens=("test",))
        result = structure.compliance(string)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_compliance_vote_aggregation(self, tokenizer):
        """Test vote aggregation."""
        models = [
            SimpleClassifier(num_classes=2, output_class=1),
            SimpleClassifier(num_classes=2, output_class=1),
        ]

        structure = MultiClassifierStructure(
            models=models,
            target_classes=[1, 1],
            tokenizer=tokenizer,
            aggregation="vote",
        )

        string = String(tokens=("test",))
        result = structure.compliance(string)

        assert isinstance(result, float)
        # Both models predict > 0.5 for target, so vote should be 1.0
        assert result == 1.0

    def test_unknown_aggregation_raises(self, models, tokenizer):
        """Test unknown aggregation raises error."""
        structure = MultiClassifierStructure(
            models=models,
            target_classes=[1, 0],
            tokenizer=tokenizer,
            aggregation="invalid",
        )

        string = String(tokens=("test",))
        with pytest.raises(ValueError, match="Unknown aggregation"):
            structure.compliance(string)
