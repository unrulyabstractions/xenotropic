"""
String representation for token sequences.

Simple wrapper around a tuple of tokens from transformer tokenizers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class String:
    """
    A sequence of tokens from a transformer tokenizer.

    Attributes:
        tokens: Tuple of token strings
        token_ids: Optional tuple of token IDs
        prompt_length: Length of prompt portion (tokens[:prompt_length] is prompt)
    """

    tokens: Tuple[str, ...]
    token_ids: Optional[Tuple[int, ...]] = None
    prompt_length: int = 0

    def __len__(self) -> int:
        return len(self.tokens)

    def __str__(self) -> str:
        return "".join(self.tokens)

    def prompt_tokens(self) -> Tuple[str, ...]:
        """Get prompt tokens."""
        return self.tokens[: self.prompt_length]

    def continuation_tokens(self) -> Tuple[str, ...]:
        """Get continuation tokens (after prompt)."""
        return self.tokens[self.prompt_length :]

    def extend(self, token: str) -> String:
        """Create new string by appending token."""
        return String(self.tokens + (token,), prompt_length=self.prompt_length)

    def extend_with_token_id(self, token: str, token_id: int) -> String:
        """Create new string by appending token with its token ID."""
        new_tokens = self.tokens + (token,)
        new_token_ids = (self.token_ids or ()) + (token_id,)
        return String(
            tokens=new_tokens, token_ids=new_token_ids, prompt_length=self.prompt_length
        )

    def to_text(self, separator: str = "") -> str:
        """Convert tokens to text."""
        return separator.join(self.tokens)

    @classmethod
    def from_text(cls, text: str) -> String:
        """Create String treating text as single token."""
        return cls(tokens=(text,))

    @classmethod
    def from_tokens(cls, tokens: Tuple[str, ...]) -> String:
        """Create String from tokens tuple."""
        return cls(tokens=tokens)

    @classmethod
    def empty(cls) -> String:
        """Create empty String (for root node)."""
        return cls(tokens=())

    def decode(self, tokenizer) -> str:
        """Decode string using tokenizer."""
        if self.token_ids:
            ids_to_decode = [tid for tid in self.token_ids if tid is not None]
            return tokenizer.decode(ids_to_decode)
        else:
            return self.to_text()
