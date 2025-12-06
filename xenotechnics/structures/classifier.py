"""
Classifier-based structures using PyTorch models.

Uses neural network models to classify strings and compute compliance
based on classification confidence.
"""

from __future__ import annotations

from typing import Callable, Optional, Union
import torch
import torch.nn as nn

from xenotechnics.common import AbstractStructure, String


def default_string_to_input(
    tokenizer,
    max_length: int = 512,
    device: Optional[Union[str, torch.device]] = None
) -> Callable[[String], torch.Tensor]:
    """
    Create a default string_to_input function using a transformers tokenizer.

    This converts a String to tokenized input suitable for transformer models.

    Args:
        tokenizer: HuggingFace tokenizer (from transformers library)
        max_length: Maximum sequence length
        device: Device to place tensors on

    Returns:
        Function that converts String → torch.Tensor

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> string_to_input = default_string_to_input(tokenizer)
        >>> structure = ClassifierStructure(
        ...     model=my_model,
        ...     target_class=1,
        ...     string_to_input=string_to_input
        ... )
    """
    def convert(string: String) -> torch.Tensor:
        # Convert String tokens to text
        text = string.to_text()

        # Tokenize using the provided tokenizer
        encoded = tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Return input_ids (most common input for classifiers)
        # Move to device if specified
        input_ids = encoded['input_ids']
        if device is not None:
            input_ids = input_ids.to(device)

        return input_ids

    return convert


class ClassifierStructure(AbstractStructure):
    """
    Structure that uses a PyTorch model to classify strings.

    The compliance is the model's confidence that the string belongs
    to the target class.

    Example use cases:
    - Sentiment compliance (positive/negative)
    - Topic compliance (on-topic/off-topic)
    - Style compliance (formal/informal)
    - Safety compliance (safe/unsafe)
    """

    def __init__(
        self,
        model: nn.Module,
        target_class: int,
        string_to_input: Optional[Callable[[String], torch.Tensor]] = None,
        tokenizer = None,
        name: str = "classifier",
        description: str = "Classifier-based structure",
        device: Optional[str] = None
    ):
        """
        Initialize ClassifierStructure.

        Args:
            model: PyTorch model that outputs class logits or probabilities
            target_class: Index of the target class for compliance
            string_to_input: Function to convert String to model input tensor.
                           If None, must provide tokenizer to use default conversion.
            tokenizer: HuggingFace tokenizer (optional, used if string_to_input is None)
            name: Structure name
            description: Structure description
            device: Device to run model on ("cpu", "cuda", etc.). If None, uses model's device.
        """
        super().__init__(name, description)
        self.model = model
        self.target_class = target_class
        self.device = device or next(model.parameters()).device

        # Set up string_to_input
        if string_to_input is None:
            if tokenizer is None:
                raise ValueError(
                    "Must provide either string_to_input or tokenizer. "
                    "Example: tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased')"
                )
            self.string_to_input = default_string_to_input(tokenizer, device=self.device)
        else:
            self.string_to_input = string_to_input

        # Set model to eval mode
        self.model.eval()

    def compliance(self, string: String) -> float:
        """
        Compute compliance as classifier confidence for target class.

        Returns:
            Probability/confidence for target_class in [0, 1]
        """
        with torch.no_grad():
            # Convert string to model input
            input_tensor = self.string_to_input(string)
            input_tensor = input_tensor.to(self.device)

            # Get model output
            output = self.model(input_tensor)

            # Apply softmax if output is logits
            if output.dim() > 1:
                # Batch dimension exists
                probs = torch.softmax(output, dim=-1)
                confidence = probs[0, self.target_class].item()
            else:
                # Single output
                probs = torch.softmax(output, dim=0)
                confidence = probs[self.target_class].item()

            return float(max(0.0, min(1.0, confidence)))


class MultiClassifierStructure(AbstractStructure):
    """
    Structure that uses multiple classifiers with voting or averaging.

    Combines predictions from multiple models to compute robust compliance.
    """

    def __init__(
        self,
        models: list[nn.Module],
        target_classes: list[int],
        string_to_input: Optional[Callable[[String], torch.Tensor]] = None,
        tokenizer = None,
        aggregation: str = "mean",
        name: str = "multi_classifier",
        description: str = "Multi-classifier ensemble",
        device: Optional[str] = None
    ):
        """
        Initialize MultiClassifierStructure.

        Args:
            models: List of PyTorch models
            target_classes: Target class index for each model
            string_to_input: Function to convert String to model input.
                           If None, must provide tokenizer to use default conversion.
            tokenizer: HuggingFace tokenizer (optional, used if string_to_input is None)
            aggregation: How to combine predictions ("mean", "max", "min", "vote")
            name: Structure name
            description: Structure description
            device: Device to run models on
        """
        super().__init__(name, description)
        self.models = models
        self.target_classes = target_classes
        self.aggregation = aggregation
        self.device = device or next(models[0].parameters()).device

        # Set up string_to_input
        if string_to_input is None:
            if tokenizer is None:
                raise ValueError(
                    "Must provide either string_to_input or tokenizer"
                )
            self.string_to_input = default_string_to_input(tokenizer, device=self.device)
        else:
            self.string_to_input = string_to_input

        # Set all models to eval mode
        for model in self.models:
            model.eval()

        if len(models) != len(target_classes):
            raise ValueError("Number of models must match number of target classes")

    def compliance(self, string: String) -> float:
        """
        Compute compliance by aggregating multiple classifier predictions.

        Returns:
            Aggregated confidence in [0, 1]
        """
        confidences = []

        with torch.no_grad():
            input_tensor = self.string_to_input(string)
            input_tensor = input_tensor.to(self.device)

            for model, target_class in zip(self.models, self.target_classes):
                output = model(input_tensor)

                if output.dim() > 1:
                    probs = torch.softmax(output, dim=-1)
                    confidence = probs[0, target_class].item()
                else:
                    probs = torch.softmax(output, dim=0)
                    confidence = probs[target_class].item()

                confidences.append(confidence)

        # Aggregate confidences
        if self.aggregation == "mean":
            result = sum(confidences) / len(confidences)
        elif self.aggregation == "max":
            result = max(confidences)
        elif self.aggregation == "min":
            result = min(confidences)
        elif self.aggregation == "vote":
            # Binary vote: count how many models predict > 0.5
            result = sum(1 for c in confidences if c > 0.5) / len(confidences)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return float(max(0.0, min(1.0, result)))
