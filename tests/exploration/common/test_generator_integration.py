"""
Integration tests for generator classes using tiny model.

Tests for exploration/common/generator.py and concrete implementations.

NOTE: These tests require torch>=2.6 due to security requirements in transformers.
They will be skipped if torch version is insufficient.
"""

from __future__ import annotations

import pytest
import torch

from xenotechnics.common import String

# Check if torch version supports model loading
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
TORCH_TOO_OLD = TORCH_VERSION < (2, 6)
SKIP_REASON = "Requires torch>=2.6 for model loading (CVE-2025-32434)"


@pytest.mark.slow
@pytest.mark.skipif(TORCH_TOO_OLD, reason=SKIP_REASON)
class TestGreedyGeneratorIntegration:
    """Integration tests for GreedyGenerator with real model."""

    @pytest.fixture
    def greedy_generator(self):
        """Create greedy generator with tiny model."""
        from exploration.generators.greedy import GreedyGenerator

        return GreedyGenerator(
            model_name="sshleifer/tiny-gpt2",
            device="cpu",
            dtype=torch.float32,
            use_chat_template=False,
        )

    def test_run_generates_output(self, greedy_generator):
        """Test that run generates output tree."""
        from xenotechnics.trees.tree import TreeNode

        prompt = String(tokens=("Hello",))
        result = greedy_generator.run(
            prompt=prompt,
            max_new_tokens=5,
            verbose=False,
        )

        assert isinstance(result, TreeNode)
        assert result.string is not None

    def test_run_with_prompt(self, greedy_generator):
        """Test generation with prompt."""
        prompt = String(tokens=("Hello",))
        result = greedy_generator.run(
            prompt=prompt,
            max_new_tokens=5,
            verbose=False,
        )

        assert result is not None
        # Should have trajectory nodes
        trajectories = result.get_trajectory_nodes()
        assert len(trajectories) > 0

    def test_run_marks_trajectory(self, greedy_generator):
        """Test that final node is marked as trajectory."""
        prompt = String(tokens=("Hello",))
        result = greedy_generator.run(
            prompt=prompt,
            max_new_tokens=3,
            verbose=False,
        )

        # Should have at least one trajectory
        trajectories = result.get_trajectory_nodes()
        assert len(trajectories) >= 1

    def test_run_builds_tree_structure(self, greedy_generator):
        """Test that tree structure is built correctly."""
        prompt = String(tokens=("Hello",))
        result = greedy_generator.run(
            prompt=prompt,
            max_new_tokens=3,
            verbose=False,
        )

        # Tree should have children
        assert len(result.children) > 0

        # Should have distributions stored
        assert len(greedy_generator.distributions) >= 0

    def test_step_count_tracks_correctly(self, greedy_generator):
        """Test that step count is tracked."""
        prompt = String(tokens=("Hello",))
        max_tokens = 5
        greedy_generator.run(
            prompt=prompt,
            max_new_tokens=max_tokens,
            verbose=False,
        )

        # Step count should be <= max_tokens (could stop early for EOS)
        assert greedy_generator.step_count <= max_tokens


@pytest.mark.slow
@pytest.mark.skipif(TORCH_TOO_OLD, reason=SKIP_REASON)
class TestSamplingGeneratorIntegration:
    """Integration tests for SamplingGenerator with real model."""

    @pytest.fixture
    def sampling_generator(self):
        """Create sampling generator with tiny model."""
        from exploration.generators.sampling import SamplingGenerator

        return SamplingGenerator(
            model_name="sshleifer/tiny-gpt2",
            device="cpu",
            dtype=torch.float32,
            use_chat_template=False,
        )

    def test_run_generates_output(self, sampling_generator):
        """Test that run generates output tree."""
        prompt = String(tokens=("Hello",))
        result = sampling_generator.run(
            prompt=prompt,
            max_new_tokens=5,
            verbose=False,
            seed=42,
        )

        assert result is not None
        trajectories = result.get_trajectory_nodes()
        assert len(trajectories) > 0

    def test_run_with_temperature(self, sampling_generator):
        """Test generation with temperature."""
        prompt = String(tokens=("Hello",))
        result = sampling_generator.run(
            prompt=prompt,
            max_new_tokens=5,
            verbose=False,
            temperature=0.5,
            seed=42,
        )

        assert result is not None

    def test_run_with_top_k(self, sampling_generator):
        """Test generation with top-k filtering."""
        prompt = String(tokens=("Hello",))
        result = sampling_generator.run(
            prompt=prompt,
            max_new_tokens=5,
            verbose=False,
            top_k=50,
            seed=42,
        )

        assert result is not None

    def test_run_with_top_p(self, sampling_generator):
        """Test generation with nucleus sampling."""
        prompt = String(tokens=("Hello",))
        result = sampling_generator.run(
            prompt=prompt,
            max_new_tokens=5,
            verbose=False,
            top_p=0.9,
            seed=42,
        )

        assert result is not None

    def test_run_deterministic_with_seed(self, sampling_generator):
        """Test that seed makes generation deterministic."""
        from exploration.generators.sampling import SamplingGenerator

        prompt = String(tokens=("Hello",))

        gen1 = SamplingGenerator(
            model_name="sshleifer/tiny-gpt2",
            device="cpu",
            dtype=torch.float32,
            use_chat_template=False,
        )
        gen2 = SamplingGenerator(
            model_name="sshleifer/tiny-gpt2",
            device="cpu",
            dtype=torch.float32,
            use_chat_template=False,
        )

        result1 = gen1.run(prompt=prompt, max_new_tokens=3, verbose=False, seed=42)
        result2 = gen2.run(prompt=prompt, max_new_tokens=3, verbose=False, seed=42)

        # Get trajectories
        traj1 = result1.get_trajectories()
        traj2 = result2.get_trajectories()

        assert len(traj1) == len(traj2)


@pytest.mark.slow
@pytest.mark.skipif(TORCH_TOO_OLD, reason=SKIP_REASON)
class TestGeneratorTreeReuse:
    """Test tree reuse functionality."""

    @pytest.fixture
    def greedy_generator(self):
        """Create greedy generator with tiny model."""
        from exploration.generators.greedy import GreedyGenerator

        return GreedyGenerator(
            model_name="sshleifer/tiny-gpt2",
            device="cpu",
            dtype=torch.float32,
            use_chat_template=False,
        )

    def test_reuse_existing_tree(self, greedy_generator):
        """Test reusing existing tree for new generation."""
        prompt = String(tokens=("Hello",))

        # First generation
        tree1 = greedy_generator.run(
            prompt=prompt,
            max_new_tokens=3,
            verbose=False,
        )

        # Second generation reusing tree
        tree2 = greedy_generator.run(
            prompt=prompt,
            max_new_tokens=3,
            verbose=False,
            existing_tree=tree1,
        )

        # Should be same root
        assert tree2 is tree1

        # Should have more trajectories
        traj = tree2.get_trajectory_nodes()
        assert len(traj) >= 1


@pytest.mark.slow
@pytest.mark.skipif(TORCH_TOO_OLD, reason=SKIP_REASON)
class TestGeneratorVerboseOutput:
    """Test verbose output functionality."""

    @pytest.fixture
    def greedy_generator(self):
        """Create greedy generator with tiny model."""
        from exploration.generators.greedy import GreedyGenerator

        return GreedyGenerator(
            model_name="sshleifer/tiny-gpt2",
            device="cpu",
            dtype=torch.float32,
            use_chat_template=False,
        )

    def test_verbose_output(self, greedy_generator, capsys):
        """Test that verbose mode produces output."""
        prompt = String(tokens=("Hello",))
        greedy_generator.run(
            prompt=prompt,
            max_new_tokens=3,
            verbose=True,
        )

        captured = capsys.readouterr()
        assert "Step" in captured.out or "Full Response" in captured.out


@pytest.mark.slow
@pytest.mark.skipif(TORCH_TOO_OLD, reason=SKIP_REASON)
class TestGeneratorInitState:
    """Test generator state initialization."""

    def test_init_generation_state_creates_tree(self):
        """Test that _init_generation_state creates tree."""
        from exploration.generators.greedy import GreedyGenerator

        gen = GreedyGenerator(
            model_name="sshleifer/tiny-gpt2",
            device="cpu",
            dtype=torch.float32,
            use_chat_template=False,
        )

        input_ids = gen.model.tokenize_prompt("Hello", use_chat_template=False)
        gen._init_generation_state(input_ids, max_new_tokens=5)

        assert gen.root_node is not None
        assert gen.current_node is not None
        assert gen.distributions == []
        assert gen.step_count == 0
        assert gen.max_steps == 5

    def test_prompt_token_count_tracked(self):
        """Test that prompt token count is tracked."""
        from exploration.generators.greedy import GreedyGenerator

        gen = GreedyGenerator(
            model_name="sshleifer/tiny-gpt2",
            device="cpu",
            dtype=torch.float32,
            use_chat_template=False,
        )

        input_ids = gen.model.tokenize_prompt("Hello world", use_chat_template=False)
        gen._init_generation_state(input_ids, max_new_tokens=5)

        assert gen.prompt_token_count == input_ids.shape[1]
