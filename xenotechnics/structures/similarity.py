"""
Embedding-based similarity structures.

Uses neural network embeddings to measure semantic similarity
between strings and target reference embeddings.
"""

from __future__ import annotations

from typing import Callable, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from xenotechnics.common import AbstractStructure, String


def default_string_to_input(
    tokenizer,
    max_length: int = 512,
    device: Optional[Union[str, torch.device]] = None
) -> Callable[[String], torch.Tensor]:
    """
    Create a default string_to_input function using a transformers tokenizer.

    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        device: Device to place tensors on

    Returns:
        Function that converts String → torch.Tensor
    """
    def convert(string: String) -> torch.Tensor:
        text = string.to_text()
        encoded = tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids']
        if device is not None:
            input_ids = input_ids.to(device)
        return input_ids

    return convert


class SimilarityStructure(AbstractStructure):
    """
    Structure that measures similarity to a reference embedding.

    Uses a model to compute embeddings and compares to a stored
    reference embedding using cosine similarity or other metrics.

    Example use cases:
    - Semantic similarity to target topic
    - Style similarity to reference text
    - Semantic coherence with prompt
    - Alignment with desired content
    """

    def __init__(
        self,
        model: nn.Module,
        reference_embedding: torch.Tensor,
        string_to_input: Optional[Callable[[String], torch.Tensor]] = None,
        tokenizer = None,
        similarity_metric: str = "cosine",
        name: str = "similarity",
        description: str = "Embedding similarity structure",
        device: Optional[str] = None
    ):
        """
        Initialize SimilarityStructure.

        Args:
            model: PyTorch model that outputs embeddings
            reference_embedding: Target embedding to compare against
            string_to_input: Function to convert String to model input.
                           If None, must provide tokenizer to use default conversion.
            tokenizer: HuggingFace tokenizer (optional, used if string_to_input is None)
            similarity_metric: Metric to use ("cosine", "euclidean", "dot")
            name: Structure name
            description: Structure description
            device: Device to run model on
        """
        super().__init__(name, description)
        self.model = model
        self.reference_embedding = reference_embedding
        self.similarity_metric = similarity_metric
        self.device = device or next(model.parameters()).device

        # Set up string_to_input
        if string_to_input is None:
            if tokenizer is None:
                raise ValueError(
                    "Must provide either string_to_input or tokenizer"
                )
            self.string_to_input = default_string_to_input(tokenizer, device=self.device)
        else:
            self.string_to_input = string_to_input

        # Move reference embedding to device
        self.reference_embedding = self.reference_embedding.to(self.device)

        # Set model to eval mode
        self.model.eval()

    def compliance(self, string: String) -> float:
        """
        Compute compliance as similarity to reference embedding.

        Returns:
            Similarity score in [0, 1]
        """
        with torch.no_grad():
            # Convert string to input and get embedding
            input_tensor = self.string_to_input(string)
            input_tensor = input_tensor.to(self.device)
            embedding = self.model(input_tensor)

            # Flatten if needed
            if embedding.dim() > 1:
                embedding = embedding.view(-1)
            if self.reference_embedding.dim() > 1:
                ref_emb = self.reference_embedding.view(-1)
            else:
                ref_emb = self.reference_embedding

            # Compute similarity
            if self.similarity_metric == "cosine":
                similarity = F.cosine_similarity(
                    embedding.unsqueeze(0),
                    ref_emb.unsqueeze(0)
                ).item()
                # Map from [-1, 1] to [0, 1]
                compliance = (similarity + 1.0) / 2.0

            elif self.similarity_metric == "dot":
                similarity = torch.dot(embedding, ref_emb).item()
                # Assume normalized embeddings, map from [-1, 1] to [0, 1]
                compliance = (similarity + 1.0) / 2.0

            elif self.similarity_metric == "euclidean":
                distance = torch.dist(embedding, ref_emb, p=2).item()
                # Convert distance to similarity (closer = higher compliance)
                # Use exponential decay: exp(-distance)
                compliance = torch.exp(-torch.tensor(distance)).item()

            else:
                raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

            return float(max(0.0, min(1.0, compliance)))

    @classmethod
    def from_reference_string(
        cls,
        model: nn.Module,
        reference_string: String,
        string_to_input: Optional[Callable[[String], torch.Tensor]] = None,
        tokenizer = None,
        similarity_metric: str = "cosine",
        name: str = "similarity",
        description: str = "Embedding similarity structure",
        device: Optional[str] = None
    ):
        """
        Create SimilarityStructure by computing embedding of reference string.

        Args:
            model: PyTorch model that outputs embeddings
            reference_string: Reference string to embed
            string_to_input: Function to convert String to model input.
                           If None, must provide tokenizer.
            tokenizer: HuggingFace tokenizer (optional, used if string_to_input is None)
            similarity_metric: Metric to use
            name: Structure name
            description: Structure description
            device: Device to run model on

        Returns:
            SimilarityStructure with reference embedding
        """
        model.eval()
        device = device or next(model.parameters()).device

        # Set up string_to_input if not provided
        if string_to_input is None:
            if tokenizer is None:
                raise ValueError("Must provide either string_to_input or tokenizer")
            string_to_input = default_string_to_input(tokenizer, device=device)

        with torch.no_grad():
            ref_input = string_to_input(reference_string).to(device)
            reference_embedding = model(ref_input)

        return cls(
            model=model,
            reference_embedding=reference_embedding,
            string_to_input=string_to_input,
            tokenizer=tokenizer,
            similarity_metric=similarity_metric,
            name=name,
            description=description,
            device=device
        )


class MultiReferenceSimilarityStructure(AbstractStructure):
    """
    Structure that measures similarity to multiple reference embeddings.

    Useful for measuring compliance with multiple valid targets or
    diverse aspects of a concept.
    """

    def __init__(
        self,
        model: nn.Module,
        reference_embeddings: list[torch.Tensor],
        string_to_input: Optional[Callable[[String], torch.Tensor]] = None,
        tokenizer = None,
        similarity_metric: str = "cosine",
        aggregation: str = "max",
        name: str = "multi_similarity",
        description: str = "Multi-reference similarity structure",
        device: Optional[str] = None
    ):
        """
        Initialize MultiReferenceSimilarityStructure.

        Args:
            model: PyTorch model that outputs embeddings
            reference_embeddings: List of target embeddings
            string_to_input: Function to convert String to model input.
                           If None, must provide tokenizer.
            tokenizer: HuggingFace tokenizer (optional, used if string_to_input is None)
            similarity_metric: Metric to use ("cosine", "euclidean", "dot")
            aggregation: How to combine similarities ("max", "mean", "min")
            name: Structure name
            description: Structure description
            device: Device to run model on
        """
        super().__init__(name, description)
        self.model = model
        self.similarity_metric = similarity_metric
        self.aggregation = aggregation
        self.device = device or next(model.parameters()).device

        # Set up string_to_input
        if string_to_input is None:
            if tokenizer is None:
                raise ValueError(
                    "Must provide either string_to_input or tokenizer"
                )
            self.string_to_input = default_string_to_input(tokenizer, device=self.device)
        else:
            self.string_to_input = string_to_input

        self.reference_embeddings = [emb.to(self.device) for emb in reference_embeddings]

        self.model.eval()

    def compliance(self, string: String) -> float:
        """
        Compute compliance as aggregated similarity to all references.

        Returns:
            Aggregated similarity in [0, 1]
        """
        with torch.no_grad():
            # Get embedding for input string
            input_tensor = self.string_to_input(string).to(self.device)
            embedding = self.model(input_tensor)

            if embedding.dim() > 1:
                embedding = embedding.view(-1)

            # Compute similarity to each reference
            similarities = []
            for ref_emb in self.reference_embeddings:
                if ref_emb.dim() > 1:
                    ref_emb = ref_emb.view(-1)

                if self.similarity_metric == "cosine":
                    sim = F.cosine_similarity(
                        embedding.unsqueeze(0),
                        ref_emb.unsqueeze(0)
                    ).item()
                    compliance_val = (sim + 1.0) / 2.0

                elif self.similarity_metric == "dot":
                    sim = torch.dot(embedding, ref_emb).item()
                    compliance_val = (sim + 1.0) / 2.0

                elif self.similarity_metric == "euclidean":
                    distance = torch.dist(embedding, ref_emb, p=2).item()
                    compliance_val = torch.exp(-torch.tensor(distance)).item()

                else:
                    raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

                similarities.append(compliance_val)

            # Aggregate
            if self.aggregation == "max":
                result = max(similarities)
            elif self.aggregation == "mean":
                result = sum(similarities) / len(similarities)
            elif self.aggregation == "min":
                result = min(similarities)
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")

            return float(max(0.0, min(1.0, result)))
