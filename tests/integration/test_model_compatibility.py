"""
Integration tests for model compatibility.

Tests that the exploration pipeline works with different model types:
- Base models vs Instruct models
- Different model sizes
- Different model families

These tests require GPU/MPS and download models, so they're marked as slow.
Run with: pytest tests/integration/test_model_compatibility.py -m slow
"""

from __future__ import annotations

import pytest

# Models to test - small ones for CI speed
MODELS = [
    pytest.param("Qwen/Qwen2.5-0.5B", id="qwen-0.5b-base"),
    pytest.param("Qwen/Qwen2.5-0.5B-Instruct", id="qwen-0.5b-instruct"),
]

# Smaller subset for quick smoke tests
SMOKE_MODELS = [
    pytest.param("Qwen/Qwen2.5-0.5B-Instruct", id="qwen-0.5b-instruct"),
]


@pytest.mark.slow
class TestModelRunnerCompatibility:
    """Test ModelRunner works with different models."""

    @pytest.fixture
    def prompt(self):
        return "Once upon a time"

    @pytest.mark.parametrize("model_name", MODELS)
    def test_model_loads(self, model_name):
        """Test that model loads without error."""
        from exploration import ModelRunner

        runner = ModelRunner(model_name)
        assert runner.model is not None

    @pytest.mark.parametrize("model_name", MODELS)
    def test_tokenize(self, model_name, prompt):
        """Test tokenization works."""
        from exploration import ModelRunner

        runner = ModelRunner(model_name)
        tokens = runner.tokenize(prompt)

        assert tokens.shape[0] == 1  # Batch size 1
        assert tokens.shape[1] > 0  # Has tokens

    @pytest.mark.parametrize("model_name", MODELS)
    def test_decode(self, model_name, prompt):
        """Test decode works."""
        from exploration import ModelRunner

        runner = ModelRunner(model_name)
        tokens = runner.tokenize(prompt)
        decoded = runner.decode(tokens[0])

        # Decoded should contain the original prompt (maybe with special tokens)
        assert "upon" in decoded.lower() or "time" in decoded.lower()

    @pytest.mark.parametrize("model_name", MODELS)
    def test_get_next_token_logits(self, model_name, prompt):
        """Test getting next token logits."""
        from exploration import ModelRunner

        runner = ModelRunner(model_name)
        tokens = runner.tokenize(prompt)
        logits, _ = runner.get_next_token_logits(tokens)

        assert logits.shape[-1] == runner.vocab_size

    @pytest.mark.parametrize("model_name", MODELS)
    def test_compute_distribution(self, model_name, prompt):
        """Test computing probability distribution from logits."""
        from exploration import ModelRunner

        runner = ModelRunner(model_name)
        tokens = runner.tokenize(prompt)
        logits, _ = runner.get_next_token_logits(tokens)
        # Logits shape may vary, get last position
        last_logits = logits[..., -1, :] if logits.dim() > 2 else logits
        last_logits = last_logits.squeeze()
        probs = runner.compute_distribution(last_logits)

        assert probs.shape[-1] == runner.vocab_size
        assert probs.sum().item() == pytest.approx(1.0, abs=1e-3)  # float16 precision
        assert (probs >= 0).all()

    @pytest.mark.parametrize("model_name", MODELS)
    def test_generate(self, model_name, prompt):
        """Test text generation."""
        from exploration import ModelRunner

        runner = ModelRunner(model_name)
        output = runner.generate(prompt, max_new_tokens=10, temperature=0.0)

        assert len(output) > 0


@pytest.mark.slow
class TestTrajectoryCollectorCompatibility:
    """Test TrajectoryCollector works with different models."""

    @pytest.fixture
    def prompt(self):
        return "The quick brown fox"

    @pytest.mark.parametrize("model_name", MODELS)
    def test_collect_trajectories(self, model_name, prompt):
        """Test collecting trajectories from different models."""
        from exploration import (
            ModelRunner,
            TrajectoryCollector,
            TrajectoryCollectorConfig,
        )

        runner = ModelRunner(model_name)
        config = TrajectoryCollectorConfig(
            max_new_tokens=5,
            temperature=1.0,
            max_iterations=10,
            target_mass=0.5,
        )
        collector = TrajectoryCollector(runner, config)

        result = collector.collect(prompt)

        assert len(result.trajectories) > 0
        assert result.total_mass > 0

        # Check trajectory structure
        for traj in result.trajectories:
            assert traj.text is not None
            assert traj.probability > 0
            assert traj.log_probability < 0  # Log of prob < 1

    @pytest.mark.parametrize("model_name", MODELS)
    def test_trajectory_reproducibility(self, model_name, prompt):
        """Test that same seed produces same trajectories."""
        from exploration import (
            ModelRunner,
            TrajectoryCollector,
            TrajectoryCollectorConfig,
        )

        runner = ModelRunner(model_name)

        config1 = TrajectoryCollectorConfig(
            max_new_tokens=5,
            temperature=1.0,
            max_iterations=5,
            seed=42,
        )
        config2 = TrajectoryCollectorConfig(
            max_new_tokens=5,
            temperature=1.0,
            max_iterations=5,
            seed=42,
        )

        result1 = TrajectoryCollector(runner, config1).collect(prompt)
        result2 = TrajectoryCollector(runner, config2).collect(prompt)

        # Same seed should produce same first trajectory
        assert result1.trajectories[0].text == result2.trajectories[0].text


@pytest.mark.slow
class TestJudgeStructureCompatibility:
    """Test JudgeStructure works with instruct models."""

    @pytest.fixture
    def text(self):
        return "The cat sat on the mat."

    @pytest.fixture
    def question(self):
        return "Does this text mention an animal?"

    # Only test instruct models - base models don't follow yes/no format
    @pytest.mark.parametrize("model_name", SMOKE_MODELS)
    def test_judge_returns_score(self, model_name, text, question):
        """Test that judge returns a valid score."""
        from exploration import ModelRunner
        from xenotechnics.structures.judge import JudgeStructure

        runner = ModelRunner(model_name)
        judge = JudgeStructure(question=question, model_runner=runner)

        score, response = judge.judge(text)

        assert 0.0 <= score <= 1.0


@pytest.mark.slow
class TestCoreEstimatorCompatibility:
    """Test CoreEstimator works with different models."""

    @pytest.mark.parametrize("model_name", SMOKE_MODELS)
    def test_full_pipeline(self, model_name):
        """Test full pipeline: collect trajectories -> estimate cores."""
        from exploration import (
            CoreEstimator,
            CoreEstimatorConfig,
            ModelRunner,
            TrajectoryCollector,
            TrajectoryCollectorConfig,
        )
        from xenotechnics.structures.judge import JudgeStructure

        # Setup
        runner = ModelRunner(model_name)
        prompt = "Once upon a time"
        structures = ["Does this mention a person?"]

        # Collect trajectories
        collector_config = TrajectoryCollectorConfig(
            max_new_tokens=5,
            temperature=1.2,
            max_iterations=10,
        )
        collector = TrajectoryCollector(runner, collector_config)
        collection_result = collector.collect(prompt)

        assert len(collection_result.trajectories) > 0

        # Estimate cores
        def make_scorer(structure):
            judge = JudgeStructure(question=structure, model_runner=runner)
            return lambda text: judge.judge(text)[0]

        estimator = CoreEstimator(CoreEstimatorConfig(use_log_space=True))
        result = estimator.estimate(
            trajectories=collection_result.trajectories,
            structures=structures,
            scorer_factory=make_scorer,
            context_prefix=prompt,
        )

        # Verify results
        assert len(result.structures) == 1
        assert 0.0 <= result.structures[0].core <= 1.0
        assert result.structures[0].expected_deviance >= 0.0


@pytest.mark.slow
class TestBaseVsInstructModels:
    """Test differences between base and instruct models."""

    def test_instruct_model_judges(self):
        """Instruct models should be able to judge text."""
        from exploration import ModelRunner
        from xenotechnics.structures.judge import JudgeStructure

        runner = ModelRunner("Qwen/Qwen2.5-0.5B-Instruct")
        judge = JudgeStructure(
            question="Does this text mention a color?",
            model_runner=runner,
        )

        # Test multiple texts
        texts = [
            "The red car drove fast.",
            "She walked to the store.",
            "The blue sky was beautiful.",
        ]

        for text in texts:
            score, _ = judge.judge(text)
            # Should get valid score
            assert 0.0 <= score <= 1.0

    def test_base_model_generates(self):
        """Base models should still generate coherent text."""
        from exploration import (
            ModelRunner,
            TrajectoryCollector,
            TrajectoryCollectorConfig,
        )

        runner = ModelRunner("Qwen/Qwen2.5-0.5B")
        config = TrajectoryCollectorConfig(
            max_new_tokens=10,
            temperature=1.0,
            max_iterations=5,
        )
        collector = TrajectoryCollector(runner, config)

        result = collector.collect("The weather today is")

        assert len(result.trajectories) > 0
        # Check we got actual text continuations
        for traj in result.trajectories:
            assert len(traj.text) > 0


@pytest.mark.slow
class TestModelSizes:
    """Test with different model sizes (when available)."""

    @pytest.mark.parametrize(
        "model_name",
        [
            pytest.param("Qwen/Qwen2.5-0.5B-Instruct", id="0.5B"),
            # Uncomment for larger models (requires more VRAM):
            # pytest.param("Qwen/Qwen2.5-1.5B-Instruct", id="1.5B"),
            # pytest.param("Qwen/Qwen2.5-3B-Instruct", id="3B"),
        ],
    )
    def test_model_size_scaling(self, model_name):
        """Test that different model sizes work."""
        from exploration import ModelRunner

        runner = ModelRunner(model_name)

        # Basic functionality check
        tokens = runner.tokenize("Hello world")
        logits, _ = runner.get_next_token_logits(tokens)

        assert logits.shape[-1] == runner.vocab_size
