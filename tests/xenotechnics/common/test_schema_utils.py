"""
Tests for schema utility classes and functions.

Tests for xenotechnics/common/schema_utils.py
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass

from xenotechnics.common import SchemaClass
from xenotechnics.common.schema_utils import (
    _canon,
    _qfloat,
    deterministic_id_from_dataclass,
)


class TestQfloat:
    """Test _qfloat function for stable float rounding."""

    def test_basic_rounding(self):
        """Test basic decimal rounding."""
        assert _qfloat(0.123456789, places=4) == 0.1235
        assert _qfloat(0.123456789, places=8) == 0.12345679

    def test_negative_zero(self):
        """Test negative zero normalizes to positive zero."""
        result = _qfloat(-0.0)
        assert result == 0.0
        # Check it's actually 0.0, not -0.0
        assert str(result) == "0.0"

    def test_small_numbers(self):
        """Test small numbers are handled correctly."""
        assert _qfloat(0.00000001, places=8) == 0.00000001
        assert _qfloat(0.000000001, places=8) == 0.0

    def test_large_numbers(self):
        """Test large numbers are handled correctly."""
        assert _qfloat(12345.6789, places=2) == 12345.68
        assert _qfloat(12345.6789, places=4) == 12345.6789

    def test_half_even_rounding(self):
        """Test banker's rounding (half-even)."""
        # 0.5 rounds to 0 (even), 1.5 rounds to 2 (even)
        assert _qfloat(0.5, places=0) == 0.0
        assert _qfloat(1.5, places=0) == 2.0
        assert _qfloat(2.5, places=0) == 2.0


class TestCanon:
    """Test _canon function for canonical representation."""

    def test_float_canonicalization(self):
        """Test floats are canonicalized."""
        result = _canon(0.123456789, places=4)
        assert result == 0.1235

    def test_nan_handling(self):
        """Test NaN is converted to string."""

        result = _canon(float("nan"))
        assert result == "NaN"

    def test_dict_canonicalization(self):
        """Test dict values are canonicalized."""
        d = {"a": 0.123456789, "b": 2}
        result = _canon(d, places=4)
        assert result == {"a": 0.1235, "b": 2}

    def test_list_canonicalization(self):
        """Test list values are canonicalized."""
        lst = [0.123456789, 0.987654321]
        result = _canon(lst, places=4)
        assert result == [0.1235, 0.9877]

    def test_tuple_canonicalization(self):
        """Test tuple values are canonicalized (becomes list)."""
        tpl = (0.123456789, 0.987654321)
        result = _canon(tpl, places=4)
        assert result == [0.1235, 0.9877]

    def test_nested_structures(self):
        """Test nested structures are fully canonicalized."""
        nested = {"outer": {"inner": [0.123456789, {"deep": 0.987654321}]}}
        result = _canon(nested, places=4)
        assert result == {"outer": {"inner": [0.1235, {"deep": 0.9877}]}}

    def test_dataclass_canonicalization(self):
        """Test dataclasses are converted to dict and canonicalized."""

        @dataclass
        class TestData:
            value: float
            name: str

        obj = TestData(value=0.123456789, name="test")
        result = _canon(obj, places=4)
        assert result == {"value": 0.1235, "name": "test"}

    def test_non_special_types_unchanged(self):
        """Test non-special types are returned unchanged."""
        assert _canon("string") == "string"
        assert _canon(42) == 42
        assert _canon(True) is True
        assert _canon(None) is None


class TestDeterministicId:
    """Test deterministic_id_from_dataclass function."""

    @dataclass
    class SimpleData:
        value: int
        name: str

    @dataclass
    class FloatData:
        x: float
        y: float

    def test_same_data_same_id(self):
        """Test identical data produces identical ID."""
        d1 = self.SimpleData(value=1, name="test")
        d2 = self.SimpleData(value=1, name="test")

        id1 = deterministic_id_from_dataclass(d1)
        id2 = deterministic_id_from_dataclass(d2)

        assert id1 == id2

    def test_different_data_different_id(self):
        """Test different data produces different ID."""
        d1 = self.SimpleData(value=1, name="test")
        d2 = self.SimpleData(value=2, name="test")

        id1 = deterministic_id_from_dataclass(d1)
        id2 = deterministic_id_from_dataclass(d2)

        assert id1 != id2

    def test_float_stability(self):
        """Test floating point values produce stable IDs."""
        d1 = self.FloatData(x=0.1 + 0.2, y=0.3)  # 0.30000000000000004
        d2 = self.FloatData(x=0.3, y=0.3)

        id1 = deterministic_id_from_dataclass(d1)
        id2 = deterministic_id_from_dataclass(d2)

        # After rounding, should be same
        assert id1 == id2

    def test_id_is_hex_string(self):
        """Test ID is a hex string of expected length."""
        d = self.SimpleData(value=1, name="test")
        id_str = deterministic_id_from_dataclass(d, digest_bytes=16)

        assert isinstance(id_str, str)
        assert len(id_str) == 32  # 16 bytes = 32 hex chars
        assert all(c in "0123456789abcdef" for c in id_str)

    def test_different_digest_sizes(self):
        """Test different digest sizes produce different length IDs."""
        d = self.SimpleData(value=1, name="test")

        id_8 = deterministic_id_from_dataclass(d, digest_bytes=8)
        id_16 = deterministic_id_from_dataclass(d, digest_bytes=16)
        id_32 = deterministic_id_from_dataclass(d, digest_bytes=32)

        assert len(id_8) == 16
        assert len(id_16) == 32
        assert len(id_32) == 64


class TestSchemaClass:
    """Test SchemaClass base class."""

    @dataclass
    class TestSchema(SchemaClass):
        name: str
        value: int
        data: list

    def test_get_id(self):
        """Test get_id returns deterministic ID."""
        s1 = self.TestSchema(name="test", value=42, data=[1, 2, 3])
        s2 = self.TestSchema(name="test", value=42, data=[1, 2, 3])

        assert s1.get_id() == s2.get_id()

    def test_get_id_different_values(self):
        """Test get_id differs for different values."""
        s1 = self.TestSchema(name="test1", value=42, data=[1, 2, 3])
        s2 = self.TestSchema(name="test2", value=42, data=[1, 2, 3])

        assert s1.get_id() != s2.get_id()

    def test_str_returns_json(self):
        """Test __str__ returns JSON representation."""
        s = self.TestSchema(name="test", value=42, data=[1, 2])
        result = str(s)

        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed["name"] == "test"
        assert parsed["value"] == 42
        assert parsed["data"] == [1, 2]

    def test_str_is_pretty_printed(self):
        """Test __str__ returns indented JSON."""
        s = self.TestSchema(name="test", value=42, data=[1, 2])
        result = str(s)

        # Should have newlines (pretty printed)
        assert "\n" in result

    def test_post_init_deep_copies(self):
        """Test __post_init__ creates deep copies of mutable fields."""
        original_list = [1, 2, 3]
        s = self.TestSchema(name="test", value=42, data=original_list)

        # Modify original
        original_list.append(4)

        # Schema should not be affected
        assert s.data == [1, 2, 3]

    def test_copy_creates_deep_copy(self):
        """Test __copy__ creates a deep copy."""
        s1 = self.TestSchema(name="test", value=42, data=[1, 2, 3])
        s2 = copy.copy(s1)

        # Should be equal but different objects
        assert s1.get_id() == s2.get_id()
        assert s1 is not s2

        # Modifying copy shouldn't affect original
        s2.data.append(4)
        assert s1.data == [1, 2, 3]

    def test_deepcopy(self):
        """Test __deepcopy__ creates a deep copy."""
        s1 = self.TestSchema(name="test", value=42, data=[1, 2, 3])
        s2 = copy.deepcopy(s1)

        assert s1.get_id() == s2.get_id()
        assert s1 is not s2
        assert s1.data is not s2.data

    def test_setattr_deep_copies(self):
        """Test __setattr__ creates deep copies on assignment."""

        @dataclass
        class MutableSchema(SchemaClass):
            items: list

        s = MutableSchema(items=[1, 2, 3])
        new_list = [4, 5, 6]
        s.items = new_list

        # Modify new_list
        new_list.append(7)

        # Schema should not be affected
        assert s.items == [4, 5, 6]


class TestSchemaClassNested:
    """Test SchemaClass with nested structures."""

    @dataclass
    class NestedSchema(SchemaClass):
        name: str
        inner: dict

    def test_nested_dict_is_copied(self):
        """Test nested dicts are deep copied."""
        inner = {"key": [1, 2, 3]}
        s = self.NestedSchema(name="test", inner=inner)

        # Modify original
        inner["key"].append(4)
        inner["new_key"] = "value"

        # Schema should not be affected
        assert s.inner == {"key": [1, 2, 3]}

    def test_nested_id_stability(self):
        """Test nested structures produce stable IDs."""
        s1 = self.NestedSchema(name="test", inner={"a": 1, "b": 2})
        s2 = self.NestedSchema(name="test", inner={"b": 2, "a": 1})

        # Dict key order shouldn't matter due to sort_keys
        assert s1.get_id() == s2.get_id()
