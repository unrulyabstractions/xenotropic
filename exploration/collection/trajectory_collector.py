"""
TrajectoryCollector for sampling trajectories from language models.

Provides a clean interface for collecting trajectories with probabilities.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch

from xenotechnics.common import SchemaClass

# Type alias for progress callback
ProgressCallback = Callable[["CollectionProgress"], None]

if TYPE_CHECKING:
    from exploration.common import ModelRunner

logger = logging.getLogger(__name__)


# Type alias for progress callback
ProgressCallback = Callable[["CollectionProgress"], None]


@dataclass
class TrajectoryCollectorConfig(SchemaClass):
    """Configuration for trajectory collection."""

    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    target_mass: float = 0.95
    max_iterations: int = 500
    max_trajectories: Optional[int] = None  # None = unlimited
    max_no_progress: int = 20
    seed: int = 42
    # Activation saving options
    save_activations: bool = False
    activation_layers: Optional[List[int]] = None  # None = all layers
    activation_components: Optional[List[str]] = None  # e.g., ["resid_post", "mlp_out"]


@dataclass
class CollectedTrajectory(SchemaClass):
    """A collected trajectory with metadata."""

    text: str
    tokens: tuple
    token_ids: tuple
    probability: float
    log_probability: float
    per_token_logprobs: List[float] = field(default_factory=list)
    is_greedy: bool = False  # True if this is the greedy (argmax) trajectory
    # Optional activation storage (dict mapping hook names to numpy arrays)
    activations: Optional[dict] = None

    def __hash__(self):
        return hash(self.token_ids)

    def __eq__(self, other):
        if not isinstance(other, CollectedTrajectory):
            return False
        return self.token_ids == other.token_ids

    def has_activations(self) -> bool:
        """Check if this trajectory has stored activations."""
        return self.activations is not None and len(self.activations) > 0


@dataclass
class CollectionProgress:
    """Progress update during collection."""

    iteration: int
    total_iterations: int
    trajectories_found: int
    total_mass: float
    target_mass: float
    elapsed_seconds: float
    no_progress_count: int

    @property
    def progress_percent(self) -> float:
        """Get progress as percentage of target mass."""
        return min(100.0, (self.total_mass / self.target_mass) * 100)

    @property
    def trajectories_per_second(self) -> float:
        """Get trajectory collection rate."""
        if self.elapsed_seconds > 0:
            return self.trajectories_found / self.elapsed_seconds
        return 0.0


@dataclass
class CollectionStats(SchemaClass):
    """Statistics from trajectory collection."""

    total_iterations: int
    unique_trajectories: int
    duplicate_trajectories: int
    failed_generations: int
    total_time_seconds: float
    avg_trajectory_length: float
    min_probability: float
    max_probability: float
    stop_reason: str  # "target_mass", "max_iterations", "no_progress"

    @property
    def trajectories_per_second(self) -> float:
        """Get trajectory collection rate."""
        if self.total_time_seconds > 0:
            return self.unique_trajectories / self.total_time_seconds
        return 0.0

    @property
    def success_rate(self) -> float:
        """Get ratio of unique trajectories to total iterations."""
        if self.total_iterations > 0:
            return self.unique_trajectories / self.total_iterations
        return 0.0


@dataclass
class CollectionResult(SchemaClass):
    """Result from trajectory collection."""

    trajectories: List[CollectedTrajectory]
    total_mass: float
    iterations: int
    stats: Optional[CollectionStats] = None

    @property
    def probabilities(self) -> np.ndarray:
        """Get array of trajectory probabilities."""
        return np.array([t.probability for t in self.trajectories])


class TrajectoryCollector:
    """
    Collects trajectories from a language model using sampling.

    Uses temperature-based sampling to explore the model's output distribution,
    collecting unique trajectories until a target probability mass is reached.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        config: Optional[TrajectoryCollectorConfig] = None,
    ):
        """
        Initialize TrajectoryCollector.

        Args:
            model_runner: ModelRunner for generation
            config: Collection configuration (uses defaults if None)
        """
        self.model_runner = model_runner
        self.config = config or TrajectoryCollectorConfig()

    def collect(
        self,
        prompt: str,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[CollectionProgress], None]] = None,
    ) -> CollectionResult:
        """
        Collect trajectories until target mass is reached.

        Args:
            prompt: Starting prompt
            seed: Random seed (uses config seed if None)
            progress_callback: Optional callback for progress updates

        Returns:
            CollectionResult with trajectories and metadata
        """
        import time

        start_time = time.time()
        trajectories = []
        total_iterations = 0
        duplicate_count = 0
        failed_count = 0
        stop_reason = "max_iterations"

        seed = seed if seed is not None else self.config.seed
        rng = np.random.default_rng(seed)

        # Tokenize prompt
        prompt_ids = self.model_runner.tokenize(prompt, prepend_bos=True)
        prompt_len = prompt_ids.shape[1]

        # Track seen trajectories
        seen_token_ids = set()
        total_mass = 0.0
        no_progress_count = 0

        # Always get greedy trajectory first
        greedy = self._greedy_trajectory(prompt_ids, prompt_len)
        if greedy is not None:
            seen_token_ids.add(greedy.token_ids)
            trajectories.append(greedy)
            total_mass += greedy.probability
            logger.debug(
                f"Greedy trajectory: p={greedy.probability:.2e}, "
                f"total_mass={total_mass:.4f}"
            )

        for iteration in range(self.config.max_iterations):
            total_iterations = iteration + 1

            # Check stopping conditions
            if total_mass >= self.config.target_mass:
                stop_reason = "target_mass"
                logger.info(f"Reached target mass {total_mass:.4f}")
                break

            if no_progress_count >= self.config.max_no_progress:
                stop_reason = "no_progress"
                logger.info(f"No progress for {no_progress_count} iterations")
                break

            if (
                self.config.max_trajectories is not None
                and len(trajectories) >= self.config.max_trajectories
            ):
                stop_reason = "max_trajectories"
                logger.info(f"Reached max trajectories {len(trajectories)}")
                break

            # Generate one trajectory
            trajectory = self._sample_trajectory(prompt_ids, prompt_len, rng)

            if trajectory is None:
                failed_count += 1
                no_progress_count += 1
                continue

            # Check if we've seen this before
            if trajectory.token_ids in seen_token_ids:
                duplicate_count += 1
                no_progress_count += 1
                continue

            # New trajectory found
            seen_token_ids.add(trajectory.token_ids)
            trajectories.append(trajectory)
            total_mass += trajectory.probability
            no_progress_count = 0

            logger.debug(
                f"Iteration {iteration}: found trajectory with p={trajectory.probability:.2e}, "
                f"total_mass={total_mass:.4f}"
            )

            # Call progress callback if provided
            if progress_callback is not None:
                elapsed = time.time() - start_time
                progress = CollectionProgress(
                    iteration=iteration,
                    total_iterations=self.config.max_iterations,
                    trajectories_found=len(trajectories),
                    total_mass=total_mass,
                    target_mass=self.config.target_mass,
                    elapsed_seconds=elapsed,
                    no_progress_count=no_progress_count,
                )
                progress_callback(progress)

        # Compute statistics
        elapsed = time.time() - start_time
        avg_length = (
            np.mean([len(t.token_ids) for t in trajectories]) if trajectories else 0.0
        )
        min_prob = min(t.probability for t in trajectories) if trajectories else 0.0
        max_prob = max(t.probability for t in trajectories) if trajectories else 0.0

        stats = CollectionStats(
            total_iterations=total_iterations,
            unique_trajectories=len(trajectories),
            duplicate_trajectories=duplicate_count,
            failed_generations=failed_count,
            total_time_seconds=elapsed,
            avg_trajectory_length=float(avg_length),
            min_probability=min_prob,
            max_probability=max_prob,
            stop_reason=stop_reason,
        )

        return CollectionResult(
            trajectories=trajectories,
            total_mass=total_mass,
            iterations=len(trajectories),
            stats=stats,
        )

    def collect_iterator(
        self, prompt: str, seed: Optional[int] = None
    ) -> Iterator[CollectedTrajectory]:
        """
        Iterate over collected trajectories.

        Yields trajectories as they are collected, useful for online processing.

        Args:
            prompt: Starting prompt
            seed: Random seed

        Yields:
            CollectedTrajectory objects
        """
        seed = seed if seed is not None else self.config.seed
        rng = np.random.default_rng(seed)

        # Tokenize prompt
        prompt_ids = self.model_runner.tokenize(prompt, prepend_bos=True)
        prompt_len = prompt_ids.shape[1]

        # Track seen trajectories
        seen_token_ids = set()
        total_mass = 0.0
        no_progress_count = 0

        for iteration in range(self.config.max_iterations):
            # Check stopping conditions
            if total_mass >= self.config.target_mass:
                logger.info(f"Reached target mass {total_mass:.4f}")
                break

            if no_progress_count >= self.config.max_no_progress:
                logger.info(f"No progress for {no_progress_count} iterations")
                break

            # Generate one trajectory
            trajectory = self._sample_trajectory(prompt_ids, prompt_len, rng)

            if trajectory is None:
                no_progress_count += 1
                continue

            # Check if we've seen this before
            if trajectory.token_ids in seen_token_ids:
                no_progress_count += 1
                continue

            # New trajectory found
            seen_token_ids.add(trajectory.token_ids)
            total_mass += trajectory.probability
            no_progress_count = 0

            logger.debug(
                f"Iteration {iteration}: found trajectory with p={trajectory.probability:.2e}, "
                f"total_mass={total_mass:.4f}"
            )

            yield trajectory

    def _greedy_trajectory(
        self,
        prompt_ids: torch.Tensor,
        prompt_len: int,
    ) -> Optional[CollectedTrajectory]:
        """
        Generate the greedy (argmax) trajectory.

        Args:
            prompt_ids: Tokenized prompt
            prompt_len: Length of prompt

        Returns:
            CollectedTrajectory with is_greedy=True, or None if generation failed
        """
        config = self.config
        device = self.model_runner.device
        eos_id = self.model_runner.eos_token_id

        generated_ids = prompt_ids.clone()
        per_token_logprobs = []
        log_prob_sum = 0.0

        with torch.no_grad():
            for step in range(config.max_new_tokens):
                logits, _ = self.model_runner.get_next_token_logits(generated_ids)
                logits = logits[0]

                # Compute log probabilities (no temperature for greedy)
                log_probs = torch.log_softmax(logits, dim=-1)

                # Pick argmax
                next_token_id = int(torch.argmax(log_probs).item())
                token_logprob = float(log_probs[next_token_id].cpu())
                per_token_logprobs.append(token_logprob)
                log_prob_sum += token_logprob

                next_token = torch.tensor([[next_token_id]], device=device)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                if eos_id is not None and next_token_id == eos_id:
                    break

        gen_ids = generated_ids[0, prompt_len:].cpu().tolist()

        if not gen_ids:
            return None

        tokens = tuple(self.model_runner.decode(torch.tensor([tid])) for tid in gen_ids)
        text = self.model_runner.decode(torch.tensor(gen_ids))

        activations = None
        if config.save_activations:
            activations = self._capture_activations(generated_ids)

        return CollectedTrajectory(
            text=text,
            tokens=tokens,
            token_ids=tuple(gen_ids),
            probability=float(np.exp(log_prob_sum)),
            log_probability=float(log_prob_sum),
            per_token_logprobs=per_token_logprobs,
            is_greedy=True,
            activations=activations,
        )

    def _sample_trajectory(
        self,
        prompt_ids: torch.Tensor,
        prompt_len: int,
        rng: np.random.Generator,
    ) -> Optional[CollectedTrajectory]:
        """
        Sample a single trajectory.

        Args:
            prompt_ids: Tokenized prompt
            prompt_len: Length of prompt
            rng: Random number generator

        Returns:
            CollectedTrajectory or None if generation failed
        """
        config = self.config
        device = self.model_runner.device
        eos_id = self.model_runner.eos_token_id

        # Initialize generation
        generated_ids = prompt_ids.clone()
        per_token_logprobs = []
        log_prob_sum = 0.0

        with torch.no_grad():
            for step in range(config.max_new_tokens):
                # Get next token distribution
                logits, _ = self.model_runner.get_next_token_logits(generated_ids)
                logits = logits[0]  # Remove batch dimension

                # Apply temperature
                if config.temperature > 0:
                    logits = logits / config.temperature

                # Compute probabilities
                probs = torch.softmax(logits, dim=-1)

                # Apply top-k filtering
                if config.top_k is not None and config.top_k > 0:
                    top_k_probs, top_k_indices = torch.topk(probs, config.top_k)
                    mask = torch.zeros_like(probs)
                    mask.scatter_(0, top_k_indices, 1.0)
                    probs = probs * mask
                    probs = probs / probs.sum()

                # Apply top-p (nucleus) filtering
                if config.top_p is not None and config.top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=0)
                    mask = cumsum <= config.top_p
                    mask[0] = True  # Always include at least one token
                    sorted_probs = sorted_probs * mask
                    probs = torch.zeros_like(probs)
                    probs.scatter_(0, sorted_indices, sorted_probs)
                    probs = probs / probs.sum()

                # Sample next token
                probs_np = probs.cpu().numpy()
                next_token_id = rng.choice(len(probs_np), p=probs_np)

                # Record log probability (use original temperature-scaled logits)
                token_logprob = float(torch.log(probs[next_token_id] + 1e-10).cpu())
                per_token_logprobs.append(token_logprob)
                log_prob_sum += token_logprob

                # Append token
                next_token = torch.tensor([[next_token_id]], device=device)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                # Check for EOS
                if eos_id is not None and next_token_id == eos_id:
                    break

        # Extract generated portion
        gen_ids = generated_ids[0, prompt_len:].cpu().tolist()

        if not gen_ids:
            return None

        # Decode tokens
        tokens = tuple(self.model_runner.decode(torch.tensor([tid])) for tid in gen_ids)
        text = self.model_runner.decode(torch.tensor(gen_ids))

        # Optionally capture activations for the full trajectory
        activations = None
        if config.save_activations:
            activations = self._capture_activations(generated_ids)

        return CollectedTrajectory(
            text=text,
            tokens=tokens,
            token_ids=tuple(gen_ids),
            probability=float(np.exp(log_prob_sum)),
            log_probability=float(log_prob_sum),
            per_token_logprobs=per_token_logprobs,
            activations=activations,
        )

    def _capture_activations(self, input_ids: torch.Tensor) -> dict:
        """
        Capture activations for the given input.

        Args:
            input_ids: Full sequence including prompt and generated tokens

        Returns:
            Dict mapping hook names to numpy arrays
        """
        config = self.config

        # Build names filter
        layers = config.activation_layers
        components = config.activation_components or ["resid_post"]

        if layers is None:
            layers = list(range(self.model_runner.n_layers))

        hook_names = set()
        for layer in layers:
            if layer < 0:
                layer = self.model_runner.n_layers + layer
            for component in components:
                hook_names.add(f"blocks.{layer}.hook_{component}")

        def names_filter(name: str) -> bool:
            return name in hook_names

        # Decode input to text for run_with_cache
        text = self.model_runner.decode(input_ids[0])

        # Run with cache
        _, cache = self.model_runner.run_with_cache(
            text,
            names_filter=names_filter,
            apply_chat_template=False,
        )

        # Convert to numpy and extract last token position
        activations = {}
        for name, tensor in cache.items():
            # tensor shape: (1, seq_len, d_model)
            # Extract last position: (d_model,)
            activations[name] = tensor[0, -1, :].cpu().numpy()

        return activations
