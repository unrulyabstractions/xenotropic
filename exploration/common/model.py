"""
Model wrapper for Transformers models.

Handles model loading and inference.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelWrapper:
    """
    Wrapper for Hugging Face transformers models.

    Handles model loading, tokenization, and inference with KV caching.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize model wrapper.

        Args:
            model_name: HuggingFace model name
            device: Device to use (auto-detected if None)
            dtype: Data type for model (auto-detected if None)
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
            if device in ["mps", "cuda"]:
                dtype = torch.float16
            else:
                dtype = torch.float32
        self.dtype = dtype

        # Load tokenizer and model
        print(f"🚀 Loading model {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device,
        )
        self.model.eval()
        print(f"✅ Model loaded: {model_name}\n")

    def tokenize_prompt(self, prompt: str, use_chat_template: bool = True) -> torch.Tensor:
        """
        Tokenize prompt.

        Args:
            prompt: Text prompt
            use_chat_template: Whether to apply chat template

        Returns:
            Token IDs tensor on model device
        """
        if use_chat_template and self.tokenizer.chat_template is not None:
            input_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            input_ids = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
            )

        return input_ids.to(self.device)

    def get_next_token_logits(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Get logits for next token with KV caching.

        Args:
            input_ids: Input token IDs
            past_key_values: KV cache from previous step

        Returns:
            Tuple of (next_token_logits, past_key_values)
        """
        with torch.no_grad():
            if past_key_values is None:
                # First step: full forward pass
                out = self.model(input_ids=input_ids, use_cache=True)
            else:
                # Subsequent steps: only process last token
                out = self.model(
                    input_ids=input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )

        logits = out.logits[:, -1, :]  # Next token logits
        past_key_values = out.past_key_values

        return logits, past_key_values

    def decode_tokens(self, token_ids: torch.Tensor, skip_special_tokens: bool = False) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def compute_distribution(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to probability distribution.

        Args:
            logits: Logits tensor

        Returns:
            Probability distribution (softmax)
        """
        return F.softmax(logits, dim=-1)

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.model.config.vocab_size

    @property
    def eos_token_id(self) -> Optional[int]:
        """Get EOS token ID."""
        return self.tokenizer.eos_token_id
