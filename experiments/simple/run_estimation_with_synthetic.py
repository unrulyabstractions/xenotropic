#!/usr/bin/env python3
"""
Run experiment with synthetic data for testing/development.

This script does EXACTLY what run_experiment.py does, but:
- Trajectories come from SyntheticTrajectoryGenerator instead of real model
- Scores come from synthetic scoring instead of real judge

Usage: python run_estimation_with_synthetic.py [trial] [--num-trajectories N]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from schemas import (
    CoreEstimationOutput,
    EstimationConfig,
    GenerationConfig,
    GenerationOutput,
    Params,
    StructureResult,
    SystemResult,
    TrajectoryRecord,
)

# -----------------------------------------------------------------------------
# Synthetic Trajectory Generator
# -----------------------------------------------------------------------------


class SyntheticTrajectoryGenerator:
    """Generate synthetic trajectories with realistic probability distributions."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        prompt_text: str,
        num_trajectories: int = 20,
    ) -> list[TrajectoryRecord]:
        """
        Generate synthetic trajectories for a prompt.

        Args:
            prompt_text: The prompt text
            num_trajectories: Number of trajectories to generate

        Returns:
            List of TrajectoryRecord with synthetic data
        """
        # Sample continuations with Zipf-like probability distribution
        continuations = [
            "Beautiful.",
            "beautiful.",
            "red.",
            "Pretty.",
            "delicate.",
            "lovely.",
            "amazing.",
            "perfect.",
            "wonderful.",
            "stunning.",
            "gorgeous.",
            "sweet.",
            "nice.",
            "fine.",
            "great.",
            "blooming.",
            "thorny.",
            "fragrant.",
            "wilting.",
            "dying.",
        ]

        # Generate Zipf-like probabilities
        ranks = np.arange(1, len(continuations) + 1)
        probs = 1.0 / (ranks**1.2)
        probs = probs / probs.sum()

        # Add some noise
        probs = probs * (1 + 0.1 * self.rng.standard_normal(len(probs)))
        probs = np.maximum(probs, 0.001)
        probs = probs / probs.sum()

        # Select trajectories
        n = min(num_trajectories, len(continuations))
        trajectories = []
        total_mass = 0.0

        for i in range(n):
            continuation = continuations[i]
            prob = float(probs[i])
            total_mass += prob

            traj = TrajectoryRecord(
                text=prompt_text + continuation,
                probability=prob,
                log_probability=float(np.log(prob + 1e-10)),
                per_token_logprobs=[],
            )
            trajectories.append(traj)

        return trajectories, total_mass


class SyntheticScorer:
    """Generate synthetic scores for trajectories."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def score(self, text: str, structure: str) -> float:
        """
        Generate a synthetic score for text given a structure question.

        Uses text content to generate deterministic but varied scores.
        """
        # Base score from text hash (deterministic)
        text_hash = hash(text + structure) % 1000
        base_score = (text_hash / 1000) * 0.6 + 0.2  # Range [0.2, 0.8]

        # Add small noise
        noise = self.rng.standard_normal() * 0.1
        score = base_score + noise

        # Bias based on keywords in structure
        text_lower = text.lower()
        if "positive" in structure.lower() or "good" in structure.lower():
            if any(
                w in text_lower
                for w in ["beautiful", "lovely", "amazing", "wonderful", "perfect"]
            ):
                score += 0.2
        if "negative" in structure.lower() or "bad" in structure.lower():
            if any(w in text_lower for w in ["dying", "wilting", "thorny"]):
                score += 0.2

        return float(np.clip(score, 0.0, 1.0))


# -----------------------------------------------------------------------------
# Helper Functions (same structure as run_experiment.py)
# -----------------------------------------------------------------------------


def collect_trajectories(
    params: Params,
    generator: SyntheticTrajectoryGenerator,
    prompt_variant: str,
    prompt_text: str,
    num_trajectories: int = 20,
    verbose: bool = True,
) -> GenerationOutput:
    """
    Generate synthetic trajectories for a single prompt.

    Args:
        params: Experiment parameters
        generator: SyntheticTrajectoryGenerator
        prompt_variant: Name of this prompt variant
        prompt_text: The actual prompt text
        num_trajectories: Number of trajectories to generate
        verbose: Print progress

    Returns:
        GenerationOutput with synthetic trajectories
    """
    if verbose:
        print(f"\n  Generating synthetic trajectories for: {prompt_variant}")
        print(f"  Prompt: {prompt_text[:60]}...")

    trajectories, total_mass = generator.generate(prompt_text, num_trajectories)

    if verbose:
        print(f"  Done: {len(trajectories)} trajectories, mass={total_mass:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return GenerationOutput(
        param_id=params.param_id,
        experiment_id=params.experiment_id,
        prompt_variant=prompt_variant,
        prompt_text=prompt_text,
        model=params.generation.model + " (synthetic)",
        timestamp=timestamp,
        total_mass=total_mass,
        num_trajectories=len(trajectories),
        trajectories=trajectories,
    )


def estimate_cores(
    params: Params,
    gen_output: GenerationOutput,
    scorer: SyntheticScorer,
    verbose: bool = True,
) -> CoreEstimationOutput:
    """
    Estimate cores for collected trajectories using CoreEstimator.

    Args:
        params: Experiment parameters
        gen_output: Generation output with trajectories
        scorer: SyntheticScorer for evaluation
        verbose: Print progress

    Returns:
        CoreEstimationOutput with core estimates
    """
    from exploration import CoreEstimator, CoreEstimatorConfig

    structures = params.estimation.structures
    system_names = params.estimation.systems

    if verbose:
        print(f"\n  Estimating cores for: {gen_output.prompt_variant}")
        print(f"  Trajectories: {gen_output.num_trajectories}")
        print(f"  Structures: {structures}")

    # Create CoreEstimator (use_log_space=False for synthetic since we have direct probs)
    estimator = CoreEstimator(CoreEstimatorConfig(use_log_space=False))

    # Create scorer factory that wraps the synthetic scorer
    def make_scorer(structure: str):
        def score_fn(text: str) -> float:
            return scorer.score(text, structure)

        return score_fn

    system_results = []

    for system_name in system_names:
        if verbose:
            print(f"\n  System: {system_name}")

        # Use CoreEstimator to compute cores
        result = estimator.estimate(
            trajectories=gen_output.trajectories,
            structures=structures,
            scorer_factory=make_scorer,
            context_prefix="",  # No context prefix for synthetic
        )

        # Convert to our schema types
        structure_results = []
        for struct_score in result.structures:
            if verbose:
                print(
                    f"    {struct_score.structure[:40]}... "
                    f"Core: {struct_score.core:.4f}, E[d]: {struct_score.expected_deviance:.4f}"
                )

            structure_results.append(
                StructureResult(
                    structure=struct_score.structure,
                    scores=struct_score.scores,
                    core=struct_score.core,
                    expected_deviance=struct_score.expected_deviance,
                    var_deviance=struct_score.var_deviance,
                )
            )

        system_results.append(
            SystemResult(
                system=system_name,
                structures=structure_results,
                aggregate_core=result.aggregate_core,
                aggregate_deviance=result.aggregate_deviance,
            )
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return CoreEstimationOutput(
        param_id=params.param_id,
        experiment_id=params.experiment_id,
        timestamp=timestamp,
        judge_model=params.estimation.model + " (synthetic)",
        prompt_variant=gen_output.prompt_variant,
        prompt_text=gen_output.prompt_text,
        num_trajectories=gen_output.num_trajectories,
        total_mass=gen_output.total_mass,
        systems=system_results,
    )


# -----------------------------------------------------------------------------
# Core Logic
# -----------------------------------------------------------------------------


def run_experiment(
    params: Params,
    num_trajectories: int = 20,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[list[GenerationOutput], list[CoreEstimationOutput]]:
    """
    Run full experiment with synthetic data.

    Args:
        params: Experiment parameters
        num_trajectories: Number of synthetic trajectories per prompt
        seed: Random seed
        verbose: Print progress

    Returns:
        Tuple of (generation_outputs, estimation_outputs)
    """
    print("=" * 70)
    print(f"EXPERIMENT (SYNTHETIC): {params.experiment_id}")
    print("=" * 70)
    print(f"Param ID: {params.param_id}")
    print(f"Generation model: {params.generation.model} (synthetic)")
    print(f"Judge model: {params.estimation.model} (synthetic)")
    print()

    # Build prompts
    base_prompt = params.generation.base_prompt
    prompts = {"base": base_prompt}

    for branch in params.generation.branching_points:
        prompts[f"branch_{branch}"] = base_prompt + branch

    print(f"Prompts ({len(prompts)}):")
    for name, prompt in prompts.items():
        print(f"  {name}: ...{prompt[-50:]}")
    print()

    # Create synthetic generators
    generator = SyntheticTrajectoryGenerator(seed=seed)
    scorer = SyntheticScorer(seed=seed + 1000)

    # Collect trajectories for each prompt
    gen_outputs = []
    for prompt_name, prompt_text in prompts.items():
        gen_output = collect_trajectories(
            params=params,
            generator=generator,
            prompt_variant=prompt_name,
            prompt_text=prompt_text,
            num_trajectories=num_trajectories,
            verbose=verbose,
        )
        gen_outputs.append(gen_output)

    # Estimate cores for each prompt
    est_outputs = []
    for gen_output in gen_outputs:
        est_output = estimate_cores(
            params=params,
            gen_output=gen_output,
            scorer=scorer,
            verbose=verbose,
        )
        est_outputs.append(est_output)

    return gen_outputs, est_outputs


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "trial",
        nargs="?",
        default="synthetic",
        help="Trial name (without .json) from trials/ directory",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=20,
        help="Number of synthetic trajectories to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    return parser.parse_args()


def load_params(trial_name: str) -> Params:
    """Load parameters from trial JSON file."""
    trials_dir = Path(__file__).parent / "trials"
    trial_path = trials_dir / f"{trial_name}.json"

    if not trial_path.exists():
        available = [p.stem for p in trials_dir.glob("*.json")]
        raise FileNotFoundError(
            f"Trial '{trial_name}' not found. Available: {available}"
        )

    with open(trial_path) as f:
        data = json.load(f)

    gen_config = GenerationConfig(**data["generation"])
    est_config = EstimationConfig(**data["estimation"])

    return Params(
        experiment_id=data["experiment_id"],
        generation=gen_config,
        estimation=est_config,
    )


def print_summary(
    gen_outputs: list[GenerationOutput],
    est_outputs: list[CoreEstimationOutput],
) -> None:
    """Print summary of results."""
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for gen_output, est_output in zip(gen_outputs, est_outputs):
        print(f"\nPrompt: {gen_output.prompt_variant}")
        print(f"  Trajectories: {gen_output.num_trajectories}")
        print(f"  Total mass: {gen_output.total_mass:.4f}")

        # Show top trajectories
        sorted_trajs = sorted(
            gen_output.trajectories,
            key=lambda t: t.probability,
            reverse=True,
        )
        print("  Top 3 trajectories:")
        for i, traj in enumerate(sorted_trajs[:3]):
            text_preview = traj.text[:50].replace("\n", "\\n")
            print(f"    {i + 1}. p={traj.probability:.4f}: {text_preview}...")

        # Show core estimates
        print("  Core estimates:")
        for sys_result in est_output.systems:
            print(f"    {sys_result.system}:")
            for struct_result in sys_result.structures:
                print(
                    f"      {struct_result.structure[:30]}... "
                    f"core={struct_result.core:.3f}, E[d]={struct_result.expected_deviance:.4f}"
                )

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE (SYNTHETIC)")
    print("=" * 70)


def main() -> int:
    args = get_args()

    # Load parameters
    params = load_params(args.trial)

    # Run experiment
    gen_outputs, est_outputs = run_experiment(
        params=params,
        num_trajectories=args.num_trajectories,
        seed=args.seed,
        verbose=not args.quiet,
    )

    # Print summary
    print_summary(gen_outputs, est_outputs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
