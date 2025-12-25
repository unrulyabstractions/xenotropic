"""
Tests for grammar-based structures.

Tests for xenotechnics/structures/grammar.py
"""

from __future__ import annotations

import pytest

from xenotechnics.common import String


class TestReadabilityStructure:
    """Test ReadabilityStructure class."""

    def test_empty_string(self):
        """Test empty string returns 1.0."""
        from xenotechnics.structures import ReadabilityStructure

        structure = ReadabilityStructure()
        assert structure.compliance(String.empty()) == 1.0
        assert structure.compliance(String.from_text("")) == 1.0

    def test_basic_readability(self):
        """Test basic readability check."""
        from xenotechnics.structures import ReadabilityStructure

        structure = ReadabilityStructure()
        string = String.from_text("The cat sat on the mat.")
        compliance = structure.compliance(string)
        assert 0.0 <= compliance <= 1.0

    def test_max_avg_word_length(self):
        """Test max average word length constraint."""
        from xenotechnics.structures import ReadabilityStructure

        structure = ReadabilityStructure(max_avg_word_length=4.0)

        # Short words
        short_words = String.from_text("The cat sat on the mat.")
        short_compliance = structure.compliance(short_words)

        # Long words
        long_words = String.from_text(
            "Supercalifragilisticexpialidocious antidisestablishmentarianism."
        )
        long_compliance = structure.compliance(long_words)

        assert short_compliance > long_compliance

    def test_max_avg_sentence_length(self):
        """Test max average sentence length constraint."""
        from xenotechnics.structures import ReadabilityStructure

        structure = ReadabilityStructure(max_avg_sentence_length=5.0)

        # Short sentence
        short = String.from_text("The cat sat.")
        short_compliance = structure.compliance(short)

        # Long sentence
        long = String.from_text(
            "The quick brown fox jumps over the lazy dog "
            "and then runs around the entire forest for hours."
        )
        long_compliance = structure.compliance(long)

        assert short_compliance > long_compliance

    def test_target_grade_level(self):
        """Test target grade level constraint."""
        from xenotechnics.structures import ReadabilityStructure

        structure = ReadabilityStructure(target_grade_level=5.0)

        # Simple text
        simple = String.from_text("The cat is big. The dog is small.")
        compliance = structure.compliance(simple)

        assert 0.0 <= compliance <= 1.0


class TestSentenceStructureStructure:
    """Test SentenceStructureStructure class."""

    def test_empty_string(self):
        """Test empty string returns 1.0."""
        from xenotechnics.structures import SentenceStructureStructure

        structure = SentenceStructureStructure()
        assert structure.compliance(String.empty()) == 1.0

    def test_proper_sentence(self):
        """Test properly structured sentence."""
        from xenotechnics.structures import SentenceStructureStructure

        structure = SentenceStructureStructure()
        string = String.from_text("The cat sat on the mat.")
        compliance = structure.compliance(string)
        assert compliance > 0.5

    def test_missing_capitalization(self):
        """Test sentence without capitalization."""
        from xenotechnics.structures import SentenceStructureStructure

        structure = SentenceStructureStructure(require_capitalization=True)
        string = String.from_text("the cat sat on the mat.")
        compliance = structure.compliance(string)
        # Should be lower due to missing capitalization
        assert compliance < 1.0

    def test_multiple_sentences(self):
        """Test multiple sentences."""
        from xenotechnics.structures import SentenceStructureStructure

        structure = SentenceStructureStructure()
        string = String.from_text("The cat sat. The dog ran. The bird flew.")
        compliance = structure.compliance(string)
        assert 0.0 <= compliance <= 1.0


class TestGrammarStructure:
    """Tests for GrammarStructure class."""

    def test_construction(self):
        """Test GrammarStructure construction."""
        from xenotechnics.structures import GrammarStructure

        structure = GrammarStructure()
        assert structure.name == "grammar"
        assert structure.language == "en-US"
        assert structure.soft is True

    def test_construction_custom_options(self):
        """Test GrammarStructure with custom options."""
        from xenotechnics.structures import GrammarStructure

        structure = GrammarStructure(language="en-GB", name="brit", soft=False)
        assert structure.name == "brit"
        assert structure.language == "en-GB"
        assert structure.soft is False

    def test_compliance_empty_string(self):
        """Test compliance with empty string returns 1.0."""
        from unittest.mock import MagicMock, PropertyMock, patch

        from xenotechnics.structures import GrammarStructure

        structure = GrammarStructure()

        # Mock the tool property
        with patch.object(
            type(structure), "tool", new_callable=PropertyMock
        ) as mock_tool:
            mock_tool.return_value = MagicMock()
            result = structure.compliance(String.empty())
            assert result == 1.0

    def test_compliance_whitespace_string(self):
        """Test compliance with whitespace string returns 1.0."""
        from unittest.mock import MagicMock, PropertyMock, patch

        from xenotechnics.structures import GrammarStructure

        structure = GrammarStructure()

        with patch.object(
            type(structure), "tool", new_callable=PropertyMock
        ) as mock_tool:
            mock_tool.return_value = MagicMock()
            result = structure.compliance(String.from_text("   "))
            assert result == 1.0

    def test_compliance_no_errors(self):
        """Test compliance with no grammar errors."""
        from unittest.mock import MagicMock, PropertyMock, patch

        from xenotechnics.structures import GrammarStructure

        structure = GrammarStructure()

        with patch.object(
            type(structure), "tool", new_callable=PropertyMock
        ) as mock_tool:
            mock_checker = MagicMock()
            mock_checker.check.return_value = []  # No errors
            mock_tool.return_value = mock_checker

            result = structure.compliance(String.from_text("Hello world."))
            assert result == 1.0

    def test_compliance_with_errors_soft(self):
        """Test compliance with errors in soft mode."""
        from unittest.mock import MagicMock, PropertyMock, patch

        from xenotechnics.structures import GrammarStructure

        structure = GrammarStructure(soft=True)

        with patch.object(
            type(structure), "tool", new_callable=PropertyMock
        ) as mock_tool:
            mock_checker = MagicMock()
            mock_checker.check.return_value = [MagicMock(), MagicMock()]  # 2 errors
            mock_tool.return_value = mock_checker

            result = structure.compliance(
                String.from_text("Hello world this is a test.")
            )
            assert 0.0 < result < 1.0

    def test_compliance_with_errors_hard(self):
        """Test compliance with errors in hard mode."""
        from unittest.mock import MagicMock, PropertyMock, patch

        from xenotechnics.structures import GrammarStructure

        structure = GrammarStructure(soft=False)

        with patch.object(
            type(structure), "tool", new_callable=PropertyMock
        ) as mock_tool:
            mock_checker = MagicMock()
            mock_checker.check.return_value = [MagicMock()]  # 1 error
            mock_tool.return_value = mock_checker

            result = structure.compliance(String.from_text("Hello world."))
            assert result == 0.0

    def test_tool_import_error(self):
        """Test tool property raises on missing dependency."""
        from unittest.mock import patch

        from xenotechnics.structures import GrammarStructure

        structure = GrammarStructure()

        with patch.dict("sys.modules", {"language_tool_python": None}):
            with pytest.raises(ImportError, match="language_tool_python not installed"):
                _ = structure.tool


class TestValidWordsStructure:
    """Tests for ValidWordsStructure class."""

    def test_construction(self):
        """Test ValidWordsStructure construction."""
        from xenotechnics.structures import ValidWordsStructure

        structure = ValidWordsStructure()
        assert structure.name == "valid_words"
        assert structure.language == "en_US"
        assert structure.ignore_markers is True
        assert structure.min_word_length == 1

    def test_construction_custom_options(self):
        """Test ValidWordsStructure with custom options."""
        from xenotechnics.structures import ValidWordsStructure

        structure = ValidWordsStructure(
            language="en_GB",
            name="brit_words",
            ignore_markers=False,
            min_word_length=3,
        )
        assert structure.name == "brit_words"
        assert structure.language == "en_GB"
        assert structure.ignore_markers is False
        assert structure.min_word_length == 3

    def test_compliance_empty_string(self):
        """Test compliance with empty string."""
        from unittest.mock import MagicMock, PropertyMock, patch

        from xenotechnics.structures import ValidWordsStructure

        structure = ValidWordsStructure()

        with patch.object(
            type(structure), "dictionary", new_callable=PropertyMock
        ) as mock_dict:
            mock_dict.return_value = MagicMock()
            result = structure.compliance(String.empty())
            assert result == 1.0

    def test_compliance_all_valid_words(self):
        """Test compliance with all valid words."""
        from unittest.mock import MagicMock, PropertyMock, patch

        from xenotechnics.structures import ValidWordsStructure

        structure = ValidWordsStructure()

        with patch.object(
            type(structure), "dictionary", new_callable=PropertyMock
        ) as mock_dict:
            mock_checker = MagicMock()
            mock_checker.check.return_value = True  # All words valid
            mock_dict.return_value = mock_checker

            result = structure.compliance(String.from_text("Hello world"))
            assert result == 1.0

    def test_compliance_some_invalid_words(self):
        """Test compliance with some invalid words."""
        from unittest.mock import MagicMock, PropertyMock, patch

        from xenotechnics.structures import ValidWordsStructure

        structure = ValidWordsStructure()

        with patch.object(
            type(structure), "dictionary", new_callable=PropertyMock
        ) as mock_dict:
            mock_checker = MagicMock()
            # First word valid, second invalid
            mock_checker.check.side_effect = [True, False]
            mock_dict.return_value = mock_checker

            result = structure.compliance(String.from_text("Hello asdfgh"))
            assert result == 0.5

    def test_compliance_ignores_markers(self):
        """Test that markers are ignored when option is set."""
        from unittest.mock import MagicMock, PropertyMock, patch

        from xenotechnics.structures import ValidWordsStructure

        structure = ValidWordsStructure(ignore_markers=True)

        with patch.object(
            type(structure), "dictionary", new_callable=PropertyMock
        ) as mock_dict:
            mock_checker = MagicMock()
            mock_checker.check.return_value = True
            mock_dict.return_value = mock_checker

            result = structure.compliance(String.from_text("Hello ⊥ world ⊤"))
            assert result == 1.0

    def test_compliance_min_word_length(self):
        """Test that short words are filtered by min_word_length."""
        from unittest.mock import MagicMock, PropertyMock, patch

        from xenotechnics.structures import ValidWordsStructure

        structure = ValidWordsStructure(min_word_length=3)

        with patch.object(
            type(structure), "dictionary", new_callable=PropertyMock
        ) as mock_dict:
            mock_checker = MagicMock()
            mock_checker.check.return_value = True
            mock_dict.return_value = mock_checker

            # "A" and "to" should be filtered out (length < 3)
            result = structure.compliance(String.from_text("A word to test"))
            assert result == 1.0

    def test_compliance_no_words(self):
        """Test compliance with text that has no valid words."""
        from unittest.mock import MagicMock, PropertyMock, patch

        from xenotechnics.structures import ValidWordsStructure

        structure = ValidWordsStructure()

        with patch.object(
            type(structure), "dictionary", new_callable=PropertyMock
        ) as mock_dict:
            mock_dict.return_value = MagicMock()
            # Only punctuation/numbers, no words
            result = structure.compliance(String.from_text("123 456 !!!"))
            assert result == 1.0

    def test_dictionary_import_error(self):
        """Test dictionary property raises on missing dependency."""
        from unittest.mock import patch

        from xenotechnics.structures import ValidWordsStructure

        structure = ValidWordsStructure()

        with patch.dict("sys.modules", {"enchant": None}):
            with pytest.raises(ImportError, match="pyenchant not installed"):
                _ = structure.dictionary


class TestPOSPatternStructure:
    """Test POSPatternStructure class.

    Requires NLTK which may or may not be installed.
    """

    def test_construction(self):
        """Test construction with patterns."""
        from xenotechnics.structures import POSPatternStructure

        # With required POS tags
        structure = POSPatternStructure(required_pos={"NN", "VB"})
        assert "NN" in structure.required_pos
        assert "VB" in structure.required_pos

        # With POS pattern
        structure2 = POSPatternStructure(pos_pattern=["DT", "NN", "VB"])
        assert structure2.pos_pattern == ["DT", "NN", "VB"]

    def test_empty_string(self):
        """Test empty string returns 1.0."""
        from xenotechnics.structures import POSPatternStructure

        structure = POSPatternStructure()
        # Empty string should return 1.0 without NLTK
        try:
            result = structure.compliance(String.empty())
            assert result == 1.0
        except ImportError:
            pytest.skip("NLTK not installed")

    def test_compliance_with_mocked_nltk(self):
        """Test compliance with mocked NLTK."""
        from unittest.mock import MagicMock, patch

        from xenotechnics.structures import POSPatternStructure

        structure = POSPatternStructure(required_pos={"NN", "VB"})

        # Create mock nltk module
        mock_nltk = MagicMock()
        mock_nltk.word_tokenize.return_value = ["The", "cat", "sits"]
        mock_nltk.pos_tag.return_value = [("The", "DT"), ("cat", "NN"), ("sits", "VB")]
        mock_nltk.data.find.return_value = True

        with patch.dict("sys.modules", {"nltk": mock_nltk}):
            result = structure.compliance(String.from_text("The cat sits"))
            assert result == 1.0  # All required POS tags found

    def test_compliance_missing_required_pos(self):
        """Test compliance when required POS tags missing."""
        from unittest.mock import MagicMock, patch

        from xenotechnics.structures import POSPatternStructure

        structure = POSPatternStructure(required_pos={"NN", "VB", "JJ"})

        mock_nltk = MagicMock()
        mock_nltk.word_tokenize.return_value = ["cat", "sits"]
        mock_nltk.pos_tag.return_value = [("cat", "NN"), ("sits", "VB")]
        mock_nltk.data.find.return_value = True

        with patch.dict("sys.modules", {"nltk": mock_nltk}):
            result = structure.compliance(String.from_text("cat sits"))
            # Missing JJ, so score should be 2/3
            assert result == pytest.approx(2 / 3)

    def test_compliance_with_pos_pattern(self):
        """Test compliance with POS pattern."""
        from unittest.mock import MagicMock, patch

        from xenotechnics.structures import POSPatternStructure

        structure = POSPatternStructure(pos_pattern=["DT", "NN"])

        mock_nltk = MagicMock()
        mock_nltk.word_tokenize.return_value = ["The", "cat"]
        mock_nltk.pos_tag.return_value = [("The", "DT"), ("cat", "NN")]
        mock_nltk.data.find.return_value = True

        with patch.dict("sys.modules", {"nltk": mock_nltk}):
            result = structure.compliance(String.from_text("The cat"))
            assert result == 1.0  # Pattern found

    def test_compliance_pos_pattern_not_found(self):
        """Test compliance when POS pattern not found."""
        from unittest.mock import MagicMock, patch

        from xenotechnics.structures import POSPatternStructure

        structure = POSPatternStructure(pos_pattern=["DT", "VB"])

        mock_nltk = MagicMock()
        mock_nltk.word_tokenize.return_value = ["The", "cat"]
        mock_nltk.pos_tag.return_value = [("The", "DT"), ("cat", "NN")]
        mock_nltk.data.find.return_value = True

        with patch.dict("sys.modules", {"nltk": mock_nltk}):
            result = structure.compliance(String.from_text("The cat"))
            assert result == 0.5  # Pattern not found, score halved

    def test_compliance_nltk_import_error(self):
        """Test compliance raises on NLTK import error."""
        from unittest.mock import patch

        from xenotechnics.structures import POSPatternStructure

        structure = POSPatternStructure()

        with patch.dict("sys.modules", {"nltk": None}):
            with pytest.raises(ImportError, match="NLTK not installed"):
                structure.compliance(String.from_text("test"))

    def test_ensure_nltk_data_downloads_missing(self):
        """Test that missing NLTK data triggers download."""
        from unittest.mock import MagicMock, patch

        from xenotechnics.structures import POSPatternStructure

        structure = POSPatternStructure()

        mock_nltk = MagicMock()
        mock_nltk.data.find.side_effect = LookupError("Not found")
        mock_nltk.download.return_value = True

        with patch.dict("sys.modules", {"nltk": mock_nltk}):
            structure._ensure_nltk_data()
            # Should have tried to download both tagger and tokenizer
            assert mock_nltk.download.call_count == 2

    def test_ensure_nltk_data_import_error(self):
        """Test ensure_nltk_data raises on import error."""
        from unittest.mock import patch

        from xenotechnics.structures import POSPatternStructure

        structure = POSPatternStructure()

        with patch.dict("sys.modules", {"nltk": None}):
            with pytest.raises(ImportError, match="NLTK not installed"):
                structure._ensure_nltk_data()
