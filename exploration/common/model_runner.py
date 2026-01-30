"""
ModelRunner for HuggingFace Transformers-based inference.

Provides a clean interface for model loading, generation, and activation capture.

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
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class ModelRunner:
    """
    Model runner for HuggingFace Transformers-based inference.

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
        """Load model using HuggingFace Transformers."""
        logger.info(f"Loading {self.model_name} on {self.device}...")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

        # Ensure pad token is set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def _detect_chat_model(self) -> bool:
        """Detect if model is instruction-tuned."""
        name = self.model_name.lower()
        # Explicit non-chat indicators (check first)
        base_indicators = ["-base", "_base"]
        if any(indicator in name for indicator in base_indicators):
            return False
        # Explicit chat indicators
        chat_indicators = ["instruct", "chat", "-it", "rlhf"]
        if any(indicator in name for indicator in chat_indicators):
            return True
        # Qwen3 models (without -Base) are instruct by default
        if "qwen3" in name:
            return True
        return False

    def _apply_chat_template(self, prompt: str) -> str:
        """Apply chat template if model is instruction-tuned."""
        if not self._is_chat_model:
            return prompt

        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                # Try with enable_thinking=False first (Qwen3 etc.)
                return self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                # Fallback for tokenizers that don't support enable_thinking
                try:
                    return self._tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    pass
            except Exception:
                pass

        # Fallback template
        return f"<|user|>\n{prompt}\n<|assistant|>\n"

    @property
    def tokenizer(self):
        """Get the tokenizer."""
        return self._tokenizer

    @property
    def n_layers(self) -> int:
        """Get number of layers."""
        return self.model.config.num_hidden_layers

    @property
    def d_model(self) -> int:
        """Get model dimension."""
        return self.model.config.hidden_size

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.model.config.vocab_size

    @property
    def eos_token_id(self) -> Optional[int]:
        """Get EOS token ID."""
        return self._tokenizer.eos_token_id

    def tokenize(self, text: str, add_special_tokens: bool = True) -> torch.Tensor:
        """
        Tokenize text.

        Args:
            text: Text to tokenize
            add_special_tokens: Whether to prepend BOS token

        Returns:
            Token IDs tensor of shape (1, seq_len)
        """
        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
            padding=False,
            truncation=False,
        )
        return encoded["input_ids"].to(self.device)

    def decode(self, token_ids: torch.Tensor) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs tensor

        Returns:
            Decoded text
        """
        if token_ids.dim() > 1:
            token_ids = token_ids[0]
        return self._tokenizer.decode(token_ids, skip_special_tokens=False)

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
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self._tokenizer.pad_token_id,
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
                outputs = self.model(input_ids[:, -1:], past_key_values=past_kv_cache)
            else:
                outputs = self.model(input_ids)

        # Return logits for last position
        return outputs.logits[:, -1, :], None

    def run_with_cache(
        self,
        prompt: str,
        names_filter: Optional[Callable[[str], bool]] = None,
        apply_chat_template: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """
        Run forward pass and capture activations using hooks.

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
        cache_dict = {}
        hooks = []

        # Register hooks on model layers to capture activations
        for layer_idx in range(self.n_layers):
            layer = self.model.model.layers[layer_idx]

            # Map TransformerLens-style names to HF module locations
            hook_points = {
                f"blocks.{layer_idx}.hook_resid_post": layer,
                f"blocks.{layer_idx}.hook_mlp_out": layer.mlp,
                f"blocks.{layer_idx}.hook_attn_out": layer.self_attn,
            }

            for name, module in hook_points.items():
                if names_filter is not None and not names_filter(name):
                    continue

                def make_hook(hook_name):
                    def hook_fn(mod, input, output):
                        # Handle different output formats
                        if isinstance(output, tuple):
                            cache_dict[hook_name] = output[0].detach()
                        else:
                            cache_dict[hook_name] = output.detach()

                    return hook_fn

                handle = module.register_forward_hook(make_hook(name))
                hooks.append(handle)

        try:
            with torch.no_grad():
                outputs = self.model(input_ids)
            logits = outputs.logits
        finally:
            # Always remove hooks
            for handle in hooks:
                handle.remove()

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
