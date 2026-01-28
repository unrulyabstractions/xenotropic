#!/usr/bin/env python3
"""
Run experiment with synthetic data for testing/development.

This script does EXACTLY what run_experiment.py does, but:
- Trajectories come from SyntheticTrajectoryGenerator instead of real model
- Scores come from synthetic scoring instead of real judge

Usage: python run_estimation_with_synthetic.py [trial] [--num-trajectories N] [--no-viz]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from schemas import (
    CoreEstimationOutput,
    GenerationOutput,
    Params,
    StructureResult,
    SystemResult,
    TrajectoryRecord,
)
from utils import (
    build_prompts,
    clean_output_dir,
    load_params,
    print_summary,
    run_visualization,
    save_outputs,
)

from exploration import CoreEstimator, CoreEstimatorConfig

# -----------------------------------------------------------------------------
# Synthetic Data Generators
# -----------------------------------------------------------------------------


class SyntheticTrajectoryGenerator:
    """Generate synthetic trajectories with realistic probability distributions."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate(self, prompt_text: str, num_trajectories: int = 20):
        """Generate synthetic trajectories for a prompt."""
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

        # Zipf-like probabilities with noise
        ranks = np.arange(1, len(continuations) + 1)
        probs = 1.0 / (ranks**1.2)
        probs = probs / probs.sum()
        probs = probs * (1 + 0.1 * self.rng.standard_normal(len(probs)))
        probs = np.maximum(probs, 0.001)
        probs = probs / probs.sum()

        n = min(num_trajectories, len(continuations))
        trajectories = []
        total_mass = 0.0

        for i in range(n):
            prob = float(probs[i])
            total_mass += prob
            trajectories.append(
                TrajectoryRecord(
                    text=prompt_text + continuations[i],
                    probability=prob,
                    log_probability=float(np.log(prob + 1e-10)),
                    per_token_logprobs=[],
                )
            )

        return trajectories, total_mass


class SyntheticScorer:
    """Generate synthetic scores for trajectories."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def score(self, text: str, structure: str) -> float:
        """Generate a synthetic score based on text and structure."""
        text_hash = hash(text + structure) % 1000
        base_score = (text_hash / 1000) * 0.6 + 0.2
        noise = self.rng.standard_normal() * 0.1
        score = base_score + noise

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
# Experiment Functions
# -----------------------------------------------------------------------------


def collect_trajectories(
    params: Params,
    generator: SyntheticTrajectoryGenerator,
    prompt_variant: str,
    prompt_text: str,
    num_trajectories: int = 20,
    verbose: bool = True,
) -> GenerationOutput:
    """Generate synthetic trajectories for a single prompt."""
    if verbose:
        print(f"\n  Generating synthetic trajectories for: {prompt_variant}")
        print(f"  Prompt: {prompt_text[:60]}...")

    trajectories, total_mass = generator.generate(prompt_text, num_trajectories)

    if verbose:
        print(f"  Done: {len(trajectories)} trajectories, mass={total_mass:.4f}")

    return GenerationOutput(
        param_id=params.param_id,
        experiment_id=params.experiment_id,
        prompt_variant=prompt_variant,
        prompt_text=prompt_text,
        model=params.generation.model + " (synthetic)",
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
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
    """Estimate cores for collected trajectories."""
    structures = params.estimation.structures
    system_names = params.estimation.systems

    if verbose:
        print(f"\n  Estimating cores for: {gen_output.prompt_variant}")
        print(f"  Trajectories: {gen_output.num_trajectories}")
        print(f"  Structures: {structures}")

    estimator = CoreEstimator(CoreEstimatorConfig(use_log_space=False))

    def make_scorer(structure: str):
        return lambda text: scorer.score(text, structure)

    system_results = []
    for system_name in system_names:
        if verbose:
            print(f"\n  System: {system_name}")

        result = estimator.estimate(
            trajectories=gen_output.trajectories,
            structures=structures,
            scorer_factory=make_scorer,
            context_prefix="",
        )

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

    return CoreEstimationOutput(
        param_id=params.param_id,
        experiment_id=params.experiment_id,
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        judge_model=params.estimation.model + " (synthetic)",
        prompt_variant=gen_output.prompt_variant,
        prompt_text=gen_output.prompt_text,
        num_trajectories=gen_output.num_trajectories,
        total_mass=gen_output.total_mass,
        systems=system_results,
    )


def run_experiment(
    params: Params,
    num_trajectories: int = 20,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[list[GenerationOutput], list[CoreEstimationOutput]]:
    """Run full experiment with synthetic data."""
    print("=" * 70)
    print(f"EXPERIMENT (SYNTHETIC): {params.experiment_id}")
    print("=" * 70)
    print(f"Param ID: {params.param_id}")
    print(f"Generation model: {params.generation.model} (synthetic)")
    print(f"Judge model: {params.estimation.model} (synthetic)")
    print()

    prompts = build_prompts(params)
    print(f"Prompts ({len(prompts)}):")
    for name, prompt in prompts.items():
        print(f"  {name}: ...{prompt[-50:]}")
    print()

    generator = SyntheticTrajectoryGenerator(seed=seed)
    scorer = SyntheticScorer(seed=seed + 1000)

    gen_outputs = []
    for prompt_name, prompt_text in prompts.items():
        gen_output = collect_trajectories(
            params, generator, prompt_name, prompt_text, num_trajectories, verbose
        )
        gen_outputs.append(gen_output)

    est_outputs = []
    for gen_output in gen_outputs:
        est_output = estimate_cores(params, gen_output, scorer, verbose)
        est_outputs.append(est_output)

    return gen_outputs, est_outputs


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "trial",
        nargs="?",
        default="test",
        help="Trial name (without .json) from trials/ directory",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=None,
        help="Number of synthetic trajectories (default: from config or 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization generation",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    return parser.parse_args()


def main() -> int:
    args = get_args()
    params = load_params(args.trial)

    num_trajectories = args.num_trajectories
    if num_trajectories is None:
        num_trajectories = params.generation.max_trajectories or 20

    output_dir = Path(__file__).parent / "out" / f"{args.trial}_synthetic"
    clean_output_dir(output_dir)
    print(f"Output directory: {output_dir}")

    gen_outputs, est_outputs = run_experiment(
        params,
        num_trajectories=num_trajectories,
        seed=args.seed,
        verbose=not args.quiet,
    )

    save_outputs(output_dir, gen_outputs, est_outputs)
    print_summary(gen_outputs, est_outputs, use_log_probs=False)

    if not args.no_viz:
        run_visualization(output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
