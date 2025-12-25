"""
Tests for String class.

Tests for xenotechnics/common/string.py
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from xenotechnics.common import String


class TestStringCreation:
    """Test String creation methods."""

    def test_create_empty_string(self):
        """Test creating empty string."""
        s = String.empty()
        assert len(s) == 0
        assert s.tokens == ()
        assert s.token_ids is None
        assert s.prompt_length == 0

    def test_create_from_text(self):
        """Test creating string from text."""
        s = String.from_text("hello world")
        assert len(s) == 1
        assert s.tokens == ("hello world",)
        assert s.token_ids is None

    def test_create_from_tokens(self):
        """Test creating string from tokens tuple."""
        tokens = ("Hello", " ", "world")
        s = String.from_tokens(tokens)
        assert len(s) == 3
        assert s.tokens == tokens

    def test_create_with_token_ids(self):
        """Test creating string with token IDs."""
        s = String(
            tokens=("a", "b", "c"),
            token_ids=(1, 2, 3),
        )
        assert s.tokens == ("a", "b", "c")
        assert s.token_ids == (1, 2, 3)

    def test_create_with_prompt_length(self):
        """Test creating string with prompt length."""
        s = String(
            tokens=("The", " ", "cat", " ", "sat"),
            prompt_length=2,
        )
        assert s.prompt_length == 2
        assert s.prompt_tokens() == ("The", " ")
        assert s.continuation_tokens() == ("cat", " ", "sat")

    def test_frozen_dataclass(self):
        """Test that String is immutable (frozen dataclass)."""
        s = String(tokens=("test",))
        with pytest.raises(AttributeError):
            s.tokens = ("new",)


class TestStringMethods:
    """Test String instance methods."""

    def test_len(self):
        """Test __len__ method."""
        assert len(String.empty()) == 0
        assert len(String(tokens=("a",))) == 1
        assert len(String(tokens=("a", "b", "c"))) == 3

    def test_str(self):
        """Test __str__ method."""
        s = String(tokens=("Hello", " ", "world"))
        assert str(s) == "Hello world"

    def test_prompt_tokens(self):
        """Test prompt_tokens method."""
        s = String(tokens=("a", "b", "c", "d"), prompt_length=2)
        assert s.prompt_tokens() == ("a", "b")

    def test_prompt_tokens_zero_length(self):
        """Test prompt_tokens with zero prompt length."""
        s = String(tokens=("a", "b", "c"))
        assert s.prompt_tokens() == ()

    def test_prompt_tokens_full_string(self):
        """Test prompt_tokens when entire string is prompt."""
        s = String(tokens=("a", "b", "c"), prompt_length=3)
        assert s.prompt_tokens() == ("a", "b", "c")

    def test_continuation_tokens(self):
        """Test continuation_tokens method."""
        s = String(tokens=("a", "b", "c", "d"), prompt_length=2)
        assert s.continuation_tokens() == ("c", "d")

    def test_continuation_tokens_zero_prompt(self):
        """Test continuation_tokens with zero prompt length."""
        s = String(tokens=("a", "b", "c"))
        assert s.continuation_tokens() == ("a", "b", "c")

    def test_continuation_tokens_full_prompt(self):
        """Test continuation_tokens when entire string is prompt."""
        s = String(tokens=("a", "b", "c"), prompt_length=3)
        assert s.continuation_tokens() == ()

    def test_extend(self):
        """Test extend method."""
        s = String(tokens=("Hello",), prompt_length=1)
        extended = s.extend(" world")
        assert extended.tokens == ("Hello", " world")
        assert extended.prompt_length == 1  # Preserved

    def test_extend_empty_string(self):
        """Test extending empty string."""
        s = String.empty()
        extended = s.extend("first")
        assert extended.tokens == ("first",)

    def test_extend_with_token_id(self):
        """Test extend_with_token_id method."""
        s = String(tokens=("a",), token_ids=(1,), prompt_length=1)
        extended = s.extend_with_token_id("b", 2)
        assert extended.tokens == ("a", "b")
        assert extended.token_ids == (1, 2)
        assert extended.prompt_length == 1

    def test_extend_with_token_id_no_existing_ids(self):
        """Test extend_with_token_id when no existing token_ids."""
        s = String(tokens=("a",))
        extended = s.extend_with_token_id("b", 2)
        assert extended.tokens == ("a", "b")
        assert extended.token_ids == (2,)

    def test_to_text_default(self):
        """Test to_text with default separator."""
        s = String(tokens=("Hello", " ", "world"))
        assert s.to_text() == "Hello world"

    def test_to_text_custom_separator(self):
        """Test to_text with custom separator."""
        s = String(tokens=("a", "b", "c"))
        assert s.to_text(separator="-") == "a-b-c"

    def test_to_text_empty(self):
        """Test to_text on empty string."""
        s = String.empty()
        assert s.to_text() == ""


class TestStringDecode:
    """Test String decode method."""

    def test_decode_with_token_ids(self):
        """Test decode with token IDs."""
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "decoded text"

        s = String(tokens=("a", "b"), token_ids=(1, 2))
        result = s.decode(tokenizer)

        tokenizer.decode.assert_called_once_with([1, 2])
        assert result == "decoded text"

    def test_decode_without_token_ids(self):
        """Test decode without token IDs falls back to to_text."""
        tokenizer = MagicMock()

        s = String(tokens=("Hello", " ", "world"))
        result = s.decode(tokenizer)

        tokenizer.decode.assert_not_called()
        assert result == "Hello world"

    def test_decode_with_none_token_ids(self):
        """Test decode filters out None token IDs."""
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "decoded"

        s = String(tokens=("a", "b", "c"), token_ids=(1, None, 3))
        result = s.decode(tokenizer)

        tokenizer.decode.assert_called_once_with([1, 3])
        assert result == "decoded"


class TestStringEquality:
    """Test String equality and hashing (frozen dataclass)."""

    def test_equality_same_tokens(self):
        """Test equality for strings with same tokens."""
        s1 = String(tokens=("a", "b", "c"))
        s2 = String(tokens=("a", "b", "c"))
        assert s1 == s2

    def test_equality_different_tokens(self):
        """Test inequality for strings with different tokens."""
        s1 = String(tokens=("a", "b", "c"))
        s2 = String(tokens=("a", "b", "d"))
        assert s1 != s2

    def test_equality_with_token_ids(self):
        """Test equality includes token_ids."""
        s1 = String(tokens=("a",), token_ids=(1,))
        s2 = String(tokens=("a",), token_ids=(1,))
        s3 = String(tokens=("a",), token_ids=(2,))
        assert s1 == s2
        assert s1 != s3

    def test_equality_with_prompt_length(self):
        """Test equality includes prompt_length."""
        s1 = String(tokens=("a", "b"), prompt_length=1)
        s2 = String(tokens=("a", "b"), prompt_length=1)
        s3 = String(tokens=("a", "b"), prompt_length=2)
        assert s1 == s2
        assert s1 != s3

    def test_hashable(self):
        """Test String is hashable (can be used in sets/dicts)."""
        s1 = String(tokens=("a", "b"))
        s2 = String(tokens=("a", "b"))
        s3 = String(tokens=("c", "d"))

        # Same content should have same hash
        assert hash(s1) == hash(s2)

        # Can be used in sets
        s = {s1, s2, s3}
        assert len(s) == 2

        # Can be used as dict keys
        d = {s1: "value1", s3: "value3"}
        assert d[s2] == "value1"


class TestStringEdgeCases:
    """Test String edge cases."""

    def test_single_token(self):
        """Test string with single token."""
        s = String(tokens=("hello",))
        assert len(s) == 1
        assert str(s) == "hello"
        assert s.to_text() == "hello"

    def test_unicode_tokens(self):
        """Test string with unicode tokens."""
        s = String(tokens=("こんにちは", "世界"))
        assert len(s) == 2
        assert str(s) == "こんにちは世界"

    def test_special_characters(self):
        """Test string with special characters."""
        s = String(tokens=("\n", "\t", "  "))
        assert len(s) == 3
        assert str(s) == "\n\t  "

    def test_empty_token_in_sequence(self):
        """Test string with empty token."""
        s = String(tokens=("a", "", "b"))
        assert len(s) == 3
        assert str(s) == "ab"

    def test_large_token_count(self):
        """Test string with many tokens."""
        tokens = tuple(f"token{i}" for i in range(1000))
        s = String(tokens=tokens)
        assert len(s) == 1000
        assert s.tokens[0] == "token0"
        assert s.tokens[999] == "token999"

    def test_prompt_length_equals_length(self):
        """Test when prompt_length equals total length."""
        s = String(tokens=("a", "b", "c"), prompt_length=3)
        assert s.prompt_tokens() == ("a", "b", "c")
        assert s.continuation_tokens() == ()

    def test_prompt_length_exceeds_length(self):
        """Test when prompt_length exceeds total length (edge case)."""
        # This is technically invalid but shouldn't crash
        s = String(tokens=("a", "b"), prompt_length=5)
        assert s.prompt_tokens() == ("a", "b")
        assert s.continuation_tokens() == ()
