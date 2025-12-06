"""
Tests for Structure classes.

Tests for xenotechnics/common/structure.py
"""

from __future__ import annotations

import pytest

from xenotechnics.common import (
    AbstractStructure,
    CompositeStructure,
    FunctionalStructure,
    String,
)


class ConcreteStructure(AbstractStructure):
    """Concrete implementation for testing."""

    def __init__(self, name: str = "test", score: float = 0.5):
        super().__init__(name, "Test structure")
        self._score = score

    def compliance(self, string: String) -> float:
        return self._score


class TestAbstractStructure:
    """Test AbstractStructure base class."""

    def test_init(self):
        """Test structure initialization."""
        s = ConcreteStructure(name="test_struct")
        assert s.name == "test_struct"
        assert s.description == "Test structure"

    def test_init_with_description(self):
        """Test initialization with custom description."""

        class DescribedStructure(AbstractStructure):
            def __init__(self):
                super().__init__("described", "Custom description")

            def compliance(self, string: String) -> float:
                return 0.5

        s = DescribedStructure()
        assert s.name == "described"
        assert s.description == "Custom description"

    def test_repr(self):
        """Test __repr__ method."""
        s = ConcreteStructure(name="my_structure")
        assert "ConcreteStructure" in repr(s)
        assert "my_structure" in repr(s)

    def test_compliance_returns_score(self):
        """Test compliance method returns expected score."""
        s = ConcreteStructure(score=0.75)
        string = String(tokens=("test",))
        assert s.compliance(string) == 0.75


class TestFunctionalStructure:
    """Test FunctionalStructure class."""

    def test_basic_compliance_function(self):
        """Test with basic compliance function."""

        def length_compliance(string: String) -> float:
            return min(1.0, len(string) / 10)

        fs = FunctionalStructure(compliance_fn=length_compliance, name="length")
        assert fs.name == "length"

        # Test compliance
        short_string = String(tokens=("a", "b"))
        long_string = String(tokens=tuple("x" for _ in range(15)))

        assert fs.compliance(short_string) == 0.2
        assert fs.compliance(long_string) == 1.0

    def test_default_name_and_description(self):
        """Test default name and description."""
        fs = FunctionalStructure(compliance_fn=lambda s: 0.5)
        assert fs.name == "custom"
        assert fs.description == "Custom compliance function"

    def test_clamps_to_zero_one(self):
        """Test compliance is clamped to [0, 1] range."""
        # Function that returns negative
        negative_fn = FunctionalStructure(compliance_fn=lambda s: -0.5)
        assert negative_fn.compliance(String.empty()) == 0.0

        # Function that returns > 1
        high_fn = FunctionalStructure(compliance_fn=lambda s: 2.0)
        assert high_fn.compliance(String.empty()) == 1.0

    def test_casts_to_float(self):
        """Test compliance converts to float."""
        # Function that returns int
        int_fn = FunctionalStructure(compliance_fn=lambda s: 1)
        result = int_fn.compliance(String.empty())
        assert isinstance(result, float)
        assert result == 1.0

    def test_with_keyword_based_compliance(self):
        """Test realistic keyword-based compliance function."""

        def keyword_compliance(string: String) -> float:
            text = string.to_text().lower()
            keywords = ["hello", "world", "test"]
            found = sum(1 for k in keywords if k in text)
            return found / len(keywords)

        fs = FunctionalStructure(
            compliance_fn=keyword_compliance,
            name="keywords",
            description="Checks for keywords",
        )

        assert fs.compliance(String(tokens=("Hello", " ", "world"))) == pytest.approx(
            2 / 3
        )
        assert fs.compliance(String(tokens=("goodbye",))) == 0.0
        assert fs.compliance(String(tokens=("hello", " ", "world", " ", "test"))) == 1.0


class TestCompositeStructure:
    """Test CompositeStructure class."""

    @pytest.fixture
    def structures(self):
        """Create test structures with known scores."""
        return [
            ConcreteStructure(name="s1", score=0.2),
            ConcreteStructure(name="s2", score=0.4),
            ConcreteStructure(name="s3", score=0.6),
            ConcreteStructure(name="s4", score=0.8),
        ]

    def test_mean_combiner(self, structures):
        """Test mean combiner."""
        cs = CompositeStructure(structures, combiner="mean", name="mean_composite")
        result = cs.compliance(String.empty())
        assert result == pytest.approx(0.5)  # (0.2 + 0.4 + 0.6 + 0.8) / 4

    def test_min_combiner(self, structures):
        """Test min combiner."""
        cs = CompositeStructure(structures, combiner="min")
        result = cs.compliance(String.empty())
        assert result == pytest.approx(0.2)

    def test_max_combiner(self, structures):
        """Test max combiner."""
        cs = CompositeStructure(structures, combiner="max")
        result = cs.compliance(String.empty())
        assert result == pytest.approx(0.8)

    def test_product_combiner(self, structures):
        """Test product combiner."""
        cs = CompositeStructure(structures, combiner="product")
        result = cs.compliance(String.empty())
        assert result == pytest.approx(0.2 * 0.4 * 0.6 * 0.8)

    def test_unknown_combiner_raises(self, structures):
        """Test unknown combiner raises ValueError."""
        cs = CompositeStructure(structures, combiner="invalid")
        with pytest.raises(ValueError, match="Unknown combiner"):
            cs.compliance(String.empty())

    def test_default_combiner_is_mean(self, structures):
        """Test default combiner is mean."""
        cs = CompositeStructure(structures)
        result = cs.compliance(String.empty())
        assert result == pytest.approx(0.5)

    def test_name_and_description(self, structures):
        """Test name and auto-generated description."""
        cs = CompositeStructure(structures, name="my_composite")
        assert cs.name == "my_composite"
        assert "4 structures" in cs.description

    def test_single_structure(self):
        """Test composite with single structure."""
        s = ConcreteStructure(score=0.7)
        cs = CompositeStructure([s], combiner="mean")
        assert cs.compliance(String.empty()) == pytest.approx(0.7)

    def test_nested_composite(self):
        """Test composite containing another composite."""
        inner = CompositeStructure(
            [ConcreteStructure(score=0.2), ConcreteStructure(score=0.4)],
            combiner="mean",
        )
        outer = CompositeStructure(
            [inner, ConcreteStructure(score=0.6)],
            combiner="mean",
        )
        # Inner gives 0.3, outer is (0.3 + 0.6) / 2 = 0.45
        result = outer.compliance(String.empty())
        assert result == pytest.approx(0.45)

    def test_product_with_zero(self):
        """Test product combiner with zero score."""
        structures = [
            ConcreteStructure(score=0.5),
            ConcreteStructure(score=0.0),
            ConcreteStructure(score=0.8),
        ]
        cs = CompositeStructure(structures, combiner="product")
        assert cs.compliance(String.empty()) == 0.0

    def test_all_same_scores(self):
        """Test with all structures returning same score."""
        structures = [ConcreteStructure(score=0.5) for _ in range(5)]
        cs = CompositeStructure(structures, combiner="mean")
        assert cs.compliance(String.empty()) == pytest.approx(0.5)


class TestStructureIntegration:
    """Integration tests for structures working together."""

    def test_functional_in_composite(self):
        """Test FunctionalStructure inside CompositeStructure."""

        def length_fn(s: String) -> float:
            return min(1.0, len(s) / 5)

        def has_vowels(s: String) -> float:
            text = s.to_text().lower()
            vowels = set("aeiou")
            if not text:
                return 0.0
            return sum(1 for c in text if c in vowels) / len(text)

        composite = CompositeStructure(
            [
                FunctionalStructure(length_fn, name="length"),
                FunctionalStructure(has_vowels, name="vowels"),
            ],
            combiner="mean",
        )

        # Test string: "hello" has 2 vowels / 5 chars = 0.4, length is 1/5 = 0.2
        string = String(tokens=("h", "e", "l", "l", "o"))
        result = composite.compliance(string)
        # length: 5/5 = 1.0, vowels: 2/5 = 0.4, mean = 0.7
        assert result == pytest.approx(0.7)

    def test_structure_accepts_any_string(self):
        """Test structures work with various string types."""
        fs = FunctionalStructure(lambda s: len(s) * 0.1, name="test")

        # Empty string
        assert fs.compliance(String.empty()) == 0.0

        # String with prompt
        s_with_prompt = String(tokens=("a", "b", "c"), prompt_length=2)
        assert fs.compliance(s_with_prompt) == pytest.approx(0.3)

        # String with token IDs
        s_with_ids = String(tokens=("x", "y"), token_ids=(1, 2))
        assert fs.compliance(s_with_ids) == pytest.approx(0.2)
