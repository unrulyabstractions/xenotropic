"""
Grammar and linguistic structures.

Structures that check for grammatical correctness, valid words,
and linguistic properties.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set

from xenotechnics.common import AbstractStructure, String


class GrammarStructure(AbstractStructure):
    """
    Structure that checks if text follows grammatical rules.

    Uses language_tool_python or similar to validate grammar.
    Requires: pip install language-tool-python
    """

    def __init__(
        self, language: str = "en-US", name: str = "grammar", soft: bool = True
    ):
        """
        Initialize GrammarStructure.

        Args:
            language: Language code (e.g., "en-US", "en-GB")
            name: Structure name
            soft: If True, gradual penalty; if False, binary 0/1
        """
        super().__init__(name, f"Grammar compliance ({language})")
        self.language = language
        self.soft = soft
        self._tool = None

    @property
    def tool(self):
        """Lazy load the grammar tool."""
        if self._tool is None:
            try:
                import language_tool_python

                self._tool = language_tool_python.LanguageTool(self.language)
            except ImportError:
                raise ImportError(
                    "language_tool_python not installed. "
                    "Install with: pip install language-tool-python"
                )
        return self._tool

    def compliance(self, string: String) -> float:
        """
        Check grammatical correctness.

        Returns:
            1.0 if no errors, decreasing with number of errors
        """
        text = string.to_text()

        # Empty text is considered correct
        if not text.strip():
            return 1.0

        matches = self.tool.check(text)
        num_errors = len(matches)

        if num_errors == 0:
            return 1.0

        if not self.soft:
            return 0.0

        # Soft: decay based on error density
        # Normalize by word count to handle varying lengths
        words = text.split()
        word_count = max(1, len(words))
        error_rate = num_errors / word_count

        # Use exponential decay
        return max(0.0, float(1.0 / (1.0 + error_rate)))


class ValidWordsStructure(AbstractStructure):
    """
    Structure that checks if all words are valid dictionary words.

    Uses enchant or NLTK wordlist for validation.
    Requires: pip install pyenchant
    """

    def __init__(
        self,
        language: str = "en_US",
        name: str = "valid_words",
        ignore_markers: bool = True,
        min_word_length: int = 1,
    ):
        """
        Initialize ValidWordsStructure.

        Args:
            language: Language code for dictionary
            name: Structure name
            ignore_markers: If True, ignore ⊥ and ⊤
            min_word_length: Minimum length to consider as word
        """
        super().__init__(name, f"Valid words ({language})")
        self.language = language
        self.ignore_markers = ignore_markers
        self.min_word_length = min_word_length
        self._dictionary = None

    @property
    def dictionary(self):
        """Lazy load the dictionary."""
        if self._dictionary is None:
            try:
                import enchant

                self._dictionary = enchant.Dict(self.language)
            except ImportError:
                raise ImportError(
                    "pyenchant not installed. Install with: pip install pyenchant"
                )
        return self._dictionary

    def compliance(self, string: String) -> float:
        """
        Check if all words are valid.

        Returns:
            Ratio of valid words to total words
        """
        text = string.to_text()

        if self.ignore_markers:
            text = text.replace("⊥", "").replace("⊤", "")

        # Extract words (alphanumeric sequences)
        words = re.findall(r"\b[a-zA-Z]+\b", text)
        words = [w for w in words if len(w) >= self.min_word_length]

        if not words:
            return 1.0

        valid_count = sum(1 for word in words if self.dictionary.check(word))
        return valid_count / len(words)


class SentenceStructureStructure(AbstractStructure):
    """
    Structure that checks for proper sentence structure.

    Validates:
    - Sentences start with capital letters
    - Sentences end with proper punctuation
    - No run-on sentences
    """

    def __init__(
        self,
        name: str = "sentence_structure",
        require_capitalization: bool = True,
        require_punctuation: bool = True,
        max_sentence_length: Optional[int] = None,
    ):
        """
        Initialize SentenceStructureStructure.

        Args:
            name: Structure name
            require_capitalization: Check for capital letters at start
            require_punctuation: Check for ending punctuation
            max_sentence_length: Maximum words per sentence
        """
        super().__init__(name, "Proper sentence structure")
        self.require_capitalization = require_capitalization
        self.require_punctuation = require_punctuation
        self.max_sentence_length = max_sentence_length

    def compliance(self, string: String) -> float:
        """
        Check sentence structure.

        Returns:
            Ratio of properly structured sentences
        """
        text = string.to_text().strip()

        if not text:
            return 1.0

        # Split into sentences (simple heuristic)
        sentence_endings = re.compile(r"[.!?]+")
        sentences = sentence_endings.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        violations = 0
        total_checks = 0

        for sentence in sentences:
            # Check capitalization
            if self.require_capitalization:
                total_checks += 1
                if not sentence[0].isupper():
                    violations += 1

            # Check punctuation (original text should have it)
            if self.require_punctuation:
                total_checks += 1
                # Look for sentence in original text with punctuation
                if not re.search(re.escape(sentence) + r"\s*[.!?]", text):
                    violations += 1

            # Check length
            if self.max_sentence_length is not None:
                total_checks += 1
                word_count = len(sentence.split())
                if word_count > self.max_sentence_length:
                    violations += 1

        if total_checks == 0:
            return 1.0

        return max(0.0, 1.0 - (violations / total_checks))


class POSPatternStructure(AbstractStructure):
    """
    Structure that checks for part-of-speech patterns.

    Uses NLTK for POS tagging and pattern matching.
    Requires: pip install nltk
    """

    def __init__(
        self,
        required_pos: Optional[Set[str]] = None,
        pos_pattern: Optional[List[str]] = None,
        name: str = "pos_pattern",
    ):
        """
        Initialize POSPatternStructure.

        Args:
            required_pos: POS tags that must appear (e.g., {"NOUN", "VERB"})
            pos_pattern: Required POS sequence (e.g., ["DET", "NOUN", "VERB"])
            name: Structure name
        """
        super().__init__(name, "POS pattern compliance")
        self.required_pos = required_pos or set()
        self.pos_pattern = pos_pattern
        self._tagger = None

    def _ensure_nltk_data(self):
        """Ensure NLTK data is downloaded."""
        try:
            import nltk

            try:
                nltk.data.find("taggers/averaged_perceptron_tagger")
            except LookupError:
                nltk.download("averaged_perceptron_tagger", quiet=True)
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)
        except ImportError:
            raise ImportError("NLTK not installed. Install with: pip install nltk")

    def compliance(self, string: String) -> float:
        """
        Check POS patterns.

        Returns:
            Compliance based on POS requirements
        """
        try:
            import nltk
        except ImportError:
            raise ImportError("NLTK not installed. Install with: pip install nltk")

        self._ensure_nltk_data()

        text = string.to_text().strip()
        if not text:
            return 1.0

        # Tokenize and tag
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)

        # Extract just the tags
        tags = [tag for _, tag in pos_tags]

        score = 1.0

        # Check required POS tags
        if self.required_pos:
            found_pos = set(tags)
            missing = self.required_pos - found_pos
            if missing:
                score *= (len(self.required_pos) - len(missing)) / len(
                    self.required_pos
                )

        # Check POS pattern
        if self.pos_pattern:
            pattern_len = len(self.pos_pattern)
            pattern_found = False

            for i in range(len(tags) - pattern_len + 1):
                if tags[i : i + pattern_len] == self.pos_pattern:
                    pattern_found = True
                    break

            if not pattern_found:
                score *= 0.5

        return float(max(0.0, min(1.0, score)))


class ReadabilityStructure(AbstractStructure):
    """
    Structure that measures text readability.

    Uses metrics like Flesch Reading Ease, average word length, etc.
    """

    def __init__(
        self,
        target_grade_level: Optional[float] = None,
        max_avg_word_length: Optional[float] = None,
        max_avg_sentence_length: Optional[float] = None,
        name: str = "readability",
    ):
        """
        Initialize ReadabilityStructure.

        Args:
            target_grade_level: Target reading grade level (0-18)
            max_avg_word_length: Maximum average word length in characters
            max_avg_sentence_length: Maximum average sentence length in words
            name: Structure name
        """
        super().__init__(name, "Readability compliance")
        self.target_grade_level = target_grade_level
        self.max_avg_word_length = max_avg_word_length
        self.max_avg_sentence_length = max_avg_sentence_length

    def compliance(self, string: String) -> float:
        """
        Check readability metrics.

        Returns:
            Compliance based on readability targets
        """
        text = string.to_text().strip()

        if not text:
            return 1.0

        # Count sentences
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = max(1, len(sentences))

        # Count words
        words = text.split()
        num_words = max(1, len(words))

        # Count syllables (rough approximation)
        def count_syllables(word):
            word = word.lower()
            count = len(re.findall(r"[aeiouy]+", word))
            if word.endswith("e"):
                count -= 1
            return max(1, count)

        num_syllables = sum(count_syllables(w) for w in words)

        score = 1.0

        # Check average word length
        if self.max_avg_word_length is not None:
            avg_word_length = sum(len(w) for w in words) / num_words
            if avg_word_length > self.max_avg_word_length:
                excess = avg_word_length - self.max_avg_word_length
                score *= max(0.0, 1.0 / (1.0 + excess))

        # Check average sentence length
        if self.max_avg_sentence_length is not None:
            avg_sentence_length = num_words / num_sentences
            if avg_sentence_length > self.max_avg_sentence_length:
                excess = avg_sentence_length - self.max_avg_sentence_length
                score *= max(0.0, 1.0 / (1.0 + excess / 10))

        # Check Flesch-Kincaid grade level
        if self.target_grade_level is not None:
            # Flesch-Kincaid Grade Level formula
            grade_level = (
                0.39 * (num_words / num_sentences)
                + 11.8 * (num_syllables / num_words)
                - 15.59
            )
            distance = abs(grade_level - self.target_grade_level)
            score *= max(0.0, 1.0 / (1.0 + distance / 5))

        return float(max(0.0, min(1.0, score)))
