"""
Tests for judge-based structures.

Tests for xenotechnics/structures/judge.py
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from xenotechnics.common import String
from xenotechnics.structures.judge import JudgeStructure


class MockTree:
    """Mock tree returned by GreedyGenerator."""

    def __init__(self, response_text: str):
        self.response_text = response_text

    def get_trajectory_nodes(self):
        node = MagicMock()
        node.string = String.from_text(self.response_text)
        return [node]


class TestJudgeStructureParseScore:
    """Test _parse_score method."""

    @patch("exploration.generators.GreedyGenerator")
    def test_parse_simple_decimal(self, mock_gen_class):
        """Test parsing simple decimal."""
        structure = JudgeStructure(question="test", model_name="test-model")
        assert structure._parse_score("0.5") == 0.5
        assert structure._parse_score("0.75") == 0.75
        assert structure._parse_score("1.0") == 1.0
        assert structure._parse_score("0.0") == 0.0

    @patch("exploration.generators.GreedyGenerator")
    def test_parse_with_whitespace(self, mock_gen_class):
        """Test parsing with whitespace."""
        structure = JudgeStructure(question="test", model_name="test-model")
        assert structure._parse_score("  0.5  ") == 0.5
        assert structure._parse_score("\n0.75\n") == 0.75

    @patch("exploration.generators.GreedyGenerator")
    def test_parse_with_text(self, mock_gen_class):
        """Test parsing score embedded in text."""
        structure = JudgeStructure(question="test", model_name="test-model")
        assert structure._parse_score("Score: 0.8") == 0.8
        assert structure._parse_score("The score is 0.6 out of 1.0") == 0.6

    @patch("exploration.generators.GreedyGenerator")
    def test_parse_integer(self, mock_gen_class):
        """Test parsing integer."""
        structure = JudgeStructure(question="test", model_name="test-model")
        assert structure._parse_score("1") == 1.0
        assert structure._parse_score("0") == 0.0

    @patch("exploration.generators.GreedyGenerator")
    def test_parse_clamps_high(self, mock_gen_class):
        """Test parsing clamps values above 100."""
        structure = JudgeStructure(question="test", model_name="test-model")
        # Values > 100 get clamped to 1.0
        assert structure._parse_score("150") == 1.0
        assert structure._parse_score("200") == 1.0

    @patch("exploration.generators.GreedyGenerator")
    def test_parse_percentage_to_decimal(self, mock_gen_class):
        """Test parsing percentage (1-100) converts to decimal."""
        structure = JudgeStructure(question="test", model_name="test-model")
        assert structure._parse_score("75") == 0.75
        assert structure._parse_score("50") == 0.50

    @patch("exploration.generators.GreedyGenerator")
    def test_parse_unparseable_returns_default(self, mock_gen_class):
        """Test unparseable string returns 0.5."""
        structure = JudgeStructure(question="test", model_name="test-model")
        # "no numbers here" returns 0.0 because it matches "no" pattern
        assert structure._parse_score("no numbers here") == 0.0
        # Empty string with no patterns returns 0.5
        assert structure._parse_score("") == 0.5
        # Random text with no numbers or patterns returns 0.5
        assert structure._parse_score("blah blah blah") == 0.5

    @patch("exploration.generators.GreedyGenerator")
    def test_parse_text_patterns(self, mock_gen_class):
        """Test parsing text patterns like 'yes', 'no'."""
        structure = JudgeStructure(question="test", model_name="test-model")
        assert structure._parse_score("no") == 0.0
        assert structure._parse_score("yes") == 1.0
        assert structure._parse_score("half") == 0.5


class TestJudgeStructureFormatPrompt:
    """Test _format_prompt method."""

    @patch("exploration.generators.GreedyGenerator")
    def test_format_prompt_includes_question(self, mock_gen_class):
        """Test prompt includes the question."""
        structure = JudgeStructure(
            question="Is this text positive?", model_name="test-model"
        )
        prompt = structure._format_prompt("Hello world")

        assert "Is this text positive?" in prompt

    @patch("exploration.generators.GreedyGenerator")
    def test_format_prompt_includes_text(self, mock_gen_class):
        """Test prompt includes the text to evaluate."""
        structure = JudgeStructure(question="test", model_name="test-model")
        prompt = structure._format_prompt("Test content here")

        assert "Test content here" in prompt

    @patch("exploration.generators.GreedyGenerator")
    def test_format_prompt_includes_instructions(self, mock_gen_class):
        """Test prompt includes scoring instructions."""
        structure = JudgeStructure(question="test", model_name="test-model")
        prompt = structure._format_prompt("text")

        assert "0" in prompt
        assert "1" in prompt


class TestJudgeStructureInit:
    """Test JudgeStructure initialization."""

    @patch("exploration.generators.GreedyGenerator")
    def test_init_basic(self, mock_gen_class):
        """Test basic initialization."""
        structure = JudgeStructure(question="Is it good?", model_name="test-model")

        assert structure.question == "Is it good?"
        assert structure.model_name == "test-model"
        assert structure._generator is None  # Lazy loaded

    @patch("exploration.generators.GreedyGenerator")
    def test_init_with_options(self, mock_gen_class):
        """Test initialization with options."""
        structure = JudgeStructure(
            question="test",
            model_name="test-model",
            device="cuda",
            use_cloud=False,
            use_chat_template=False,
            isolate=True,
        )

        assert structure.device == "cuda"
        assert structure.use_cloud is False
        assert structure.use_chat_template is False
        assert structure.isolate is True

    @patch("exploration.generators.GreedyGenerator")
    def test_name_includes_question(self, mock_gen_class):
        """Test that name includes truncated question."""
        structure = JudgeStructure(
            question="Is this a very long question that should be truncated?",
            model_name="test-model",
        )

        assert "Judge:" in structure.name
        assert len(structure.name) < 100  # Should be truncated

    @patch("exploration.generators.GreedyGenerator")
    def test_description_includes_question(self, mock_gen_class):
        """Test that description includes question."""
        structure = JudgeStructure(question="Is it good?", model_name="test-model")

        assert "Is it good?" in structure.description


class TestJudgeStructureGenerator:
    """Test generator lazy loading."""

    def test_generator_lazy_loads_local(self):
        """Test generator lazy loads GreedyGenerator for local."""
        with patch("exploration.generators.GreedyGenerator") as mock_gen_class:
            mock_gen = MagicMock()
            mock_gen_class.return_value = mock_gen

            structure = JudgeStructure(question="test", model_name="test-model")
            assert structure._generator is None

            # Access generator
            gen = structure.generator
            assert gen == mock_gen
            mock_gen_class.assert_called_once()

    def test_generator_lazy_loads_cloud(self):
        """Test generator lazy loads CloudGreedyGenerator for cloud."""
        with patch("exploration.generators.CloudGreedyGenerator") as mock_gen_class:
            mock_gen = MagicMock()
            mock_gen_class.return_value = mock_gen

            structure = JudgeStructure(
                question="test", model_name="test-model", use_cloud=True
            )
            assert structure._generator is None

            # Access generator
            gen = structure.generator
            assert gen == mock_gen
            mock_gen_class.assert_called_once_with("test-model")


class TestJudgeStructureCompliance:
    """Test JudgeStructure compliance method."""

    def test_compliance_returns_float(self):
        """Test compliance returns a float."""
        with patch("exploration.generators.GreedyGenerator") as mock_gen_class:
            mock_gen = MagicMock()
            # Return tree with "0.8" as the response
            prompt_text = JudgeStructure.PROMPT_TEMPLATE.format(
                question="Is it good?", text="test text"
            )
            mock_gen.run.return_value = MockTree(prompt_text + "0.8")
            mock_gen_class.return_value = mock_gen

            structure = JudgeStructure(question="Is it good?", model_name="test-model")
            string = String(tokens=("test", " ", "text"))
            result = structure.compliance(string)

            assert isinstance(result, float)
            assert result == 0.8

    def test_compliance_in_range(self):
        """Test compliance is in [0, 1]."""
        with patch("exploration.generators.GreedyGenerator") as mock_gen_class:
            mock_gen = MagicMock()
            mock_gen.run.return_value = MockTree("prefix0.5")
            mock_gen_class.return_value = mock_gen

            structure = JudgeStructure(question="Is it good?", model_name="test-model")

            test_strings = [
                String(tokens=("hello",)),
                String(tokens=("world",)),
                String(tokens=("test", " ", "string")),
            ]

            for s in test_strings:
                result = structure.compliance(s)
                assert 0.0 <= result <= 1.0

    def test_compliance_empty_string(self):
        """Test compliance with empty string."""
        with patch("exploration.generators.GreedyGenerator") as mock_gen_class:
            mock_gen = MagicMock()
            mock_gen.run.return_value = MockTree("prefix0.5")
            mock_gen_class.return_value = mock_gen

            structure = JudgeStructure(question="test", model_name="test-model")

            string = String.empty()
            result = structure.compliance(string)

            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0


class TestJudgeStructureJudge:
    """Test JudgeStructure judge method."""

    def test_judge_returns_tuple(self):
        """Test judge returns (score, response) tuple."""
        with patch("exploration.generators.GreedyGenerator") as mock_gen_class:
            mock_gen = MagicMock()
            prompt_text = JudgeStructure.PROMPT_TEMPLATE.format(
                question="test", text="hello"
            )
            mock_gen.run.return_value = MockTree(prompt_text + "0.75")
            mock_gen_class.return_value = mock_gen

            structure = JudgeStructure(question="test", model_name="test-model")
            score, response = structure.judge("hello")

            assert isinstance(score, float)
            assert score == 0.75
            assert isinstance(response, str)
            assert "0.75" in response

    def test_judge_cloud(self):
        """Test judge with cloud generator."""
        with patch("exploration.generators.CloudGreedyGenerator") as mock_gen_class:
            mock_gen = MagicMock()
            mock_result = MagicMock()
            mock_result.text = "  0.9  "
            mock_gen.generate.return_value = mock_result
            mock_gen_class.return_value = mock_gen

            structure = JudgeStructure(
                question="test", model_name="test-model", use_cloud=True
            )
            score, response = structure.judge("hello")

            assert score == 0.9
            assert response == "0.9"


class TestJudgeStructureQueryLocal:
    """Test _query_local method."""

    def test_query_local_extracts_continuation(self):
        """Test _query_local extracts continuation from tree."""
        with patch("exploration.generators.GreedyGenerator") as mock_gen_class:
            mock_gen = MagicMock()
            # Full response = prompt + continuation
            mock_gen.run.return_value = MockTree("my prompt0.65")
            mock_gen_class.return_value = mock_gen

            structure = JudgeStructure(question="test", model_name="test-model")
            result = structure._query_local("my prompt")

            assert result == "0.65"

    def test_query_local_empty_trajectory(self):
        """Test _query_local with empty trajectory."""
        with patch("exploration.generators.GreedyGenerator") as mock_gen_class:
            mock_gen = MagicMock()
            mock_tree = MagicMock()
            mock_tree.get_trajectory_nodes.return_value = []
            mock_gen.run.return_value = mock_tree
            mock_gen_class.return_value = mock_gen

            structure = JudgeStructure(question="test", model_name="test-model")
            result = structure._query_local("prompt")

            assert result == ""


class TestJudgeStructureQueryCloud:
    """Test _query_cloud method."""

    def test_query_cloud_success(self):
        """Test _query_cloud success."""
        with patch("exploration.generators.CloudGreedyGenerator") as mock_gen_class:
            mock_gen = MagicMock()
            mock_result = MagicMock()
            mock_result.text = "  0.85  "
            mock_gen.generate.return_value = mock_result
            mock_gen_class.return_value = mock_gen

            structure = JudgeStructure(
                question="test", model_name="test-model", use_cloud=True
            )
            result = structure._query_cloud("prompt")

            assert result == "0.85"

    def test_query_cloud_error(self):
        """Test _query_cloud handles errors."""
        with patch("exploration.generators.CloudGreedyGenerator") as mock_gen_class:
            mock_gen = MagicMock()
            mock_gen.generate.side_effect = Exception("API Error")
            mock_gen_class.return_value = mock_gen

            structure = JudgeStructure(
                question="test", model_name="test-model", use_cloud=True
            )
            result = structure._query_cloud("prompt")

            assert "[Error:" in result
