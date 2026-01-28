"""
Tests for JudgeStructure.

Tests for xenotechnics/structures/judge.py
"""

from __future__ import annotations

import pytest

from xenotechnics.common import String
from xenotechnics.structures.judge import JudgeStructure


class MockModelRunner:
    """Mock ModelRunner for testing."""

    def __init__(self, response: str = "0.75"):
        self.response = response
        self.generate_calls = []

    def generate(
        self, prompt, max_new_tokens=20, temperature=0.0, apply_chat_template=True
    ):
        self.generate_calls.append(prompt)
        return self.response


class TestJudgeStructureInit:
    """Test JudgeStructure initialization."""

    def test_init_with_model_runner(self):
        """Test initialization with model runner."""
        mock_runner = MockModelRunner()
        judge = JudgeStructure(
            question="Is this text positive?",
            model_runner=mock_runner,
        )

        assert judge.question == "Is this text positive?"
        assert judge._model_runner is mock_runner

    def test_init_with_model_name(self):
        """Test initialization with model name (lazy loading)."""
        judge = JudgeStructure(
            question="Is this text positive?",
            model_name="test-model",
        )

        assert judge.question == "Is this text positive?"
        assert judge._model_name == "test-model"
        assert judge._model_runner is None  # Not loaded yet

    def test_init_sets_name_and_description(self):
        """Test that name and description are set."""
        mock_runner = MockModelRunner()
        judge = JudgeStructure(
            question="Is this text positive?",
            model_runner=mock_runner,
        )

        assert "Judge:" in judge.name
        assert "positive" in judge.name
        assert "positive" in judge.description

    def test_name_truncates_long_question(self):
        """Test that name truncates long questions."""
        mock_runner = MockModelRunner()
        long_question = "A" * 100
        judge = JudgeStructure(
            question=long_question,
            model_runner=mock_runner,
        )

        assert len(judge.name) < 100
        assert "..." in judge.name


class TestJudgeStructureModelRunnerProperty:
    """Test model_runner property."""

    def test_returns_provided_runner(self):
        """Test returns provided runner."""
        mock_runner = MockModelRunner()
        judge = JudgeStructure(question="test", model_runner=mock_runner)

        assert judge.model_runner is mock_runner

    def test_raises_without_name_or_runner(self):
        """Test raises error if no runner or model name."""
        judge = JudgeStructure(question="test")

        with pytest.raises(ValueError, match="Must provide"):
            _ = judge.model_runner


class TestJudgeStructureCompliance:
    """Test JudgeStructure.compliance()."""

    def test_compliance_calls_generate(self):
        """Test that compliance calls generate."""
        mock_runner = MockModelRunner(response="0.8")
        judge = JudgeStructure(
            question="Is this text positive?",
            model_runner=mock_runner,
        )

        string = String(tokens=("Hello", " ", "world"))
        score = judge.compliance(string)

        assert len(mock_runner.generate_calls) == 1
        assert "Hello world" in mock_runner.generate_calls[0]
        assert score == pytest.approx(0.8)

    def test_compliance_includes_question_in_prompt(self):
        """Test that question is included in prompt."""
        mock_runner = MockModelRunner(response="0.5")
        judge = JudgeStructure(
            question="Does this contain cats?",
            model_runner=mock_runner,
        )

        string = String(tokens=("The", " ", "cat", " ", "sat"))
        judge.compliance(string)

        assert "Does this contain cats?" in mock_runner.generate_calls[0]

    def test_compliance_empty_string(self):
        """Test compliance with empty string."""
        mock_runner = MockModelRunner(response="0.5")
        judge = JudgeStructure(question="test", model_runner=mock_runner)

        result = judge.compliance(String.empty())

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


class TestJudgeStructureJudge:
    """Test JudgeStructure.judge()."""

    def test_judge_returns_score_and_response(self):
        """Test that judge returns both score and response."""
        mock_runner = MockModelRunner(response="0.9")
        judge = JudgeStructure(
            question="Is this positive?",
            model_runner=mock_runner,
        )

        score, response = judge.judge("Great day!")

        assert score == pytest.approx(0.9)
        assert response == "0.9"

    def test_judge_includes_text_in_prompt(self):
        """Test that judge includes text in prompt."""
        mock_runner = MockModelRunner(response="0.5")
        judge = JudgeStructure(question="test", model_runner=mock_runner)

        judge.judge("Sample text here")

        assert "Sample text here" in mock_runner.generate_calls[0]


class TestJudgeStructureParseScore:
    """Test JudgeStructure._parse_score()."""

    @pytest.fixture
    def judge(self):
        """Create judge for testing."""
        mock_runner = MockModelRunner()
        return JudgeStructure(question="test", model_runner=mock_runner)

    @pytest.mark.parametrize(
        "response,expected",
        [
            # Direct decimals
            ("0.5", 0.5),
            ("0.0", 0.0),
            ("1.0", 1.0),
            ("0.75", 0.75),
            (".5", 0.5),
            # Integers
            ("0", 0.0),
            ("1", 1.0),
            # Percentages (converted)
            ("50", 0.5),
            ("75", 0.75),
            ("100", 1.0),
            # With whitespace
            ("  0.8  ", 0.8),
            ("\n0.6\n", 0.6),
            # Numbers in text
            ("Score: 0.7", 0.7),
            ("The answer is 0.4", 0.4),
            # Text patterns
            ("no", 0.0),
            ("NOT at all", 0.0),
            ("yes", 1.0),
            ("completely", 1.0),
            ("half", 0.5),
            # Edge cases - default to 0.5
            ("maybe", 0.5),
            ("", 0.5),
            ("unknown", 0.5),
        ],
    )
    def test_parse_score(self, judge, response, expected):
        """Test score parsing with various inputs."""
        result = judge._parse_score(response)
        assert result == pytest.approx(expected)

    def test_parse_score_clamps_high(self, judge):
        """Test that scores > 1 are clamped."""
        result = judge._parse_score("1.5")
        assert result == 1.0

    def test_parse_score_clamps_low(self, judge):
        """Test that scores < 0 are clamped."""
        result = judge._parse_score("-0.5")
        assert result == 0.0

    def test_parse_score_percentage_over_100(self, judge):
        """Test that percentages > 100 are clamped."""
        result = judge._parse_score("150")
        assert result == 1.0
