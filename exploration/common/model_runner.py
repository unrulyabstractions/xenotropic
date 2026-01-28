"""
ModelRunner for TransformerLens-based inference.

Provides a clean interface for model loading, generation, and activation capture.
Follows patterns from temporal-awareness but simplified for TransformerLens only.

Example:
    runner = ModelRunner("Qwen/Qwen2.5-0.5B-Instruct")
    output = runner.generate("What is 2+2?")

    # With activation capture
    logits, cache = runner.run_with_cache("Hello world")
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch

logger = logging.getLogger(__name__)


class ModelRunner:
    """
    Model runner for TransformerLens-based inference.

    Provides:
    - Model loading with automatic device/dtype detection
    - Text generation with temperature control
    - Activation caching for interpretability
    - Chat template handling for instruct models
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize ModelRunner.

        Args:
            model_name: HuggingFace model name or path
            device: Device to use (auto-detected if None: MPS > CUDA > CPU)
            dtype: Data type (auto-detected if None: float16 for GPU, float32 for CPU)
        """
        self.model_name = model_name

        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        # Auto-detect dtype
        if dtype is None:
            dtype = torch.float16 if device in ["mps", "cuda"] else torch.float32
        self.dtype = dtype

        # Load model
        self._load_model()

        # Detect chat model
        self._is_chat_model = self._detect_chat_model()

        logger.info(
            f"ModelRunner initialized: {model_name} on {device} "
            f"(chat={self._is_chat_model}, n_layers={self.n_layers}, d_model={self.d_model})"
        )

    def _load_model(self) -> None:
        """Load model using TransformerLens."""
        from transformer_lens import HookedTransformer

        logger.info(f"Loading {self.model_name} on {self.device}...")

        self.model = HookedTransformer.from_pretrained_no_processing(
            self.model_name,
            device=self.device,
            dtype=self.dtype,
        )
        self.model.eval()

    def _detect_chat_model(self) -> bool:
        """Detect if model is instruction-tuned."""
        name = self.model_name.lower()
        chat_indicators = ["instruct", "chat", "-it", "rlhf"]
        return any(indicator in name for indicator in chat_indicators)

    def _apply_chat_template(self, prompt: str) -> str:
        """Apply chat template if model is instruction-tuned."""
        if not self._is_chat_model:
            return prompt

        tokenizer = self.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        # Fallback template
        return f"<|user|>\n{prompt}\n<|assistant|>\n"

    @property
    def tokenizer(self):
        """Get the tokenizer."""
        return self.model.tokenizer

    @property
    def n_layers(self) -> int:
        """Get number of layers."""
        return self.model.cfg.n_layers

    @property
    def d_model(self) -> int:
        """Get model dimension."""
        return self.model.cfg.d_model

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.model.cfg.d_vocab

    @property
    def eos_token_id(self) -> Optional[int]:
        """Get EOS token ID."""
        return self.tokenizer.eos_token_id

    def tokenize(self, text: str, prepend_bos: bool = True) -> torch.Tensor:
        """
        Tokenize text.

        Args:
            text: Text to tokenize
            prepend_bos: Whether to prepend BOS token

        Returns:
            Token IDs tensor of shape (1, seq_len)
        """
        return self.model.to_tokens(text, prepend_bos=prepend_bos)

    def decode(self, token_ids: torch.Tensor) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs tensor

        Returns:
            Decoded text
        """
        return self.model.to_string(token_ids)

    def compute_distribution(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute probability distribution from logits.

        Args:
            logits: Logits tensor of shape (vocab_size,) or (batch, vocab_size)

        Returns:
            Probability distribution (same shape as input)
        """
        return torch.softmax(logits, dim=-1)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        apply_chat_template: bool = True,
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            apply_chat_template: Whether to apply chat template for instruct models

        Returns:
            Generated text (excluding prompt)
        """
        if apply_chat_template:
            formatted = self._apply_chat_template(prompt)
        else:
            formatted = prompt

        input_ids = self.tokenize(formatted)
        prompt_len = input_ids.shape[1]

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "stop_at_eos": True,
            "verbose": False,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **gen_kwargs)

        return self.decode(output_ids[0, prompt_len:])

    def get_next_token_logits(
        self,
        input_ids: torch.Tensor,
        past_kv_cache: Any = None,
    ) -> tuple[torch.Tensor, Any]:
        """
        Get logits for next token.

        Args:
            input_ids: Input token IDs of shape (1, seq_len)
            past_kv_cache: Optional KV cache from previous call

        Returns:
            Tuple of (logits, new_kv_cache) where logits has shape (1, vocab_size)
        """
        with torch.no_grad():
            if past_kv_cache is not None:
                # Only pass last token when using cache
                logits = self.model(input_ids[:, -1:], past_kv_cache=past_kv_cache)
            else:
                logits = self.model(input_ids)

        # Return logits for last position
        return logits[
            :, -1, :
        ], None  # TransformerLens doesn't return KV cache directly

    def run_with_cache(
        self,
        prompt: str,
        names_filter: Optional[Callable[[str], bool]] = None,
        apply_chat_template: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """
        Run forward pass and capture activations.

        Args:
            prompt: Input prompt
            names_filter: Function to filter which hooks to cache
                         (e.g., lambda name: "resid_post" in name)
            apply_chat_template: Whether to apply chat template

        Returns:
            Tuple of (logits, cache) where cache is a dict of activations
        """
        if apply_chat_template:
            formatted = self._apply_chat_template(prompt)
        else:
            formatted = prompt

        input_ids = self.tokenize(formatted)

        with torch.no_grad():
            logits, cache = self.model.run_with_cache(
                input_ids,
                names_filter=names_filter,
            )

        # Convert ActivationCache to dict
        cache_dict = {name: cache[name] for name in cache.keys()}

        return logits, cache_dict

    def get_activation_names(
        self,
        layers: Optional[list[int]] = None,
        components: list[str] = None,
    ) -> list[str]:
        """
        Get hook names for specified layers and components.

        Args:
            layers: Layer indices (None = all layers)
            components: Components to capture (default: ["resid_post"])

        Returns:
            List of hook names like "blocks.0.hook_resid_post"
        """
        if layers is None:
            layers = list(range(self.n_layers))
        if components is None:
            components = ["resid_post"]

        names = []
        for layer in layers:
            # Handle negative indices
            if layer < 0:
                layer = self.n_layers + layer
            for component in components:
                names.append(f"blocks.{layer}.hook_{component}")

        return names

    def __repr__(self) -> str:
        return (
            f"ModelRunner({self.model_name}, device={self.device}, "
            f"n_layers={self.n_layers}, d_model={self.d_model})"
        )
