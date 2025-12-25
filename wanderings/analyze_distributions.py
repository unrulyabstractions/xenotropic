"""
Helper script to load and analyze full token distributions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# -----------------------------------------------------------------------------
# Input/Output Data Structures
# -----------------------------------------------------------------------------


@dataclass
class AnalyzeDistributionsInput:
    """Input for distribution analysis."""

    output_dir: Path


@dataclass
class StepAnalysis:
    """Analysis for a single generation step."""

    step: int
    sampled_token: str
    sampled_probability: float
    entropy: float
    min_prob: float
    max_prob: float
    non_zero_tokens: int


@dataclass
class AnalyzeDistributionsOutput:
    """Output from distribution analysis."""

    model: str
    prompt: str
    num_steps: int
    shape: tuple[int, int]
    step_analyses: list[StepAnalysis]


# -----------------------------------------------------------------------------
# Core Logic
# -----------------------------------------------------------------------------


def load_distributions(output_dir: Path) -> tuple[np.ndarray, dict]:
    """Load the full distributions and metadata."""
    with open(output_dir / "token_distributions_metadata.json") as f:
        metadata = json.load(f)

    data = np.load(output_dir / "token_distributions_full.npz")
    distributions = data["distributions"]

    return distributions, metadata


def analyze_step(
    distributions: np.ndarray, metadata: dict, step_num: int
) -> StepAnalysis:
    """Analyze a specific generation step."""
    step_dist = distributions[step_num]
    step_meta = metadata["distributions"][step_num]

    return StepAnalysis(
        step=step_num,
        sampled_token=step_meta["sampled_token"],
        sampled_probability=step_meta["sampled_probability"],
        entropy=step_meta["entropy"],
        min_prob=float(step_dist.min()),
        max_prob=float(step_dist.max()),
        non_zero_tokens=int(np.count_nonzero(step_dist)),
    )


def analyze_distributions(inp: AnalyzeDistributionsInput) -> AnalyzeDistributionsOutput:
    """Main analysis logic."""
    distributions, metadata = load_distributions(inp.output_dir)

    step_analyses = []
    steps_to_analyze = (
        [0, 5, 10] if len(distributions) > 10 else range(len(distributions))
    )

    for step in steps_to_analyze:
        if step < len(distributions):
            step_analyses.append(analyze_step(distributions, metadata, step))

    return AnalyzeDistributionsOutput(
        model=metadata["model"],
        prompt=metadata["prompt"],
        num_steps=metadata["num_steps"],
        shape=distributions.shape,
        step_analyses=step_analyses,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "output",
        help="Directory containing distribution files",
    )
    return parser.parse_args()


def input_from_args(args: argparse.Namespace) -> AnalyzeDistributionsInput:
    """Load input from command line arguments."""
    return AnalyzeDistributionsInput(output_dir=args.output_dir)


def save_output(args: argparse.Namespace, output: AnalyzeDistributionsOutput) -> None:
    """Save output (no-op for this script)."""
    pass


def print_output(args: argparse.Namespace, output: AnalyzeDistributionsOutput) -> None:
    """Print output to stdout."""
    print("Loaded distributions")
    print(f"   Shape: {output.shape}")
    print(f"   Model: {output.model}")
    print(f"   Prompt: {output.prompt}")
    print(f"   Generated: {output.num_steps} tokens")

    for analysis in output.step_analyses:
        print(f"\n{'=' * 60}")
        print(f"Step {analysis.step}")
        print(f"{'=' * 60}")
        print(f"Sampled token: '{analysis.sampled_token}'")
        print(f"Sampled probability: {analysis.sampled_probability:.6f}")
        print(f"Entropy: {analysis.entropy:.4f}")
        print("\nDistribution stats:")
        print(f"  Min probability: {analysis.min_prob:.2e}")
        print(f"  Max probability: {analysis.max_prob:.6f}")
        print(f"  Non-zero tokens: {analysis.non_zero_tokens:,}")

    print("\n" + "=" * 60)
    print("Analysis complete!")


def main() -> int:
    args = get_args()
    inp: AnalyzeDistributionsInput = input_from_args(args)
    output: AnalyzeDistributionsOutput = analyze_distributions(inp)

    save_output(args, output)
    print_output(args, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
