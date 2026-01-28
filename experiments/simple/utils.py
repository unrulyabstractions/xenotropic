"""Shared utilities for experiment scripts."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

from schemas import (
    CoreEstimationOutput,
    EstimationConfig,
    GenerationConfig,
    GenerationOutput,
    Params,
)


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


def clean_output_dir(output_dir: Path) -> None:
    """Remove all files in output directory."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def save_outputs(
    output_dir: Path,
    gen_outputs: list[GenerationOutput],
    est_outputs: list[CoreEstimationOutput],
) -> None:
    """Save generation and estimation outputs to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for gen_output in gen_outputs:
        filename = f"gen_{gen_output.prompt_variant}.json"
        filepath = output_dir / filename
        with open(filepath, "w") as f:
            json.dump(asdict(gen_output), f, indent=2)
        print(f"Saved: {filepath}")

    for est_output in est_outputs:
        filename = f"est_{est_output.prompt_variant}.json"
        filepath = output_dir / filename
        with open(filepath, "w") as f:
            json.dump(asdict(est_output), f, indent=2)
        print(f"Saved: {filepath}")


def print_summary(
    gen_outputs: list[GenerationOutput],
    est_outputs: list[CoreEstimationOutput],
    use_log_probs: bool = True,
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
        if use_log_probs:
            sorted_trajs = sorted(
                gen_output.trajectories,
                key=lambda t: t.log_probability,
                reverse=True,
            )
        else:
            sorted_trajs = sorted(
                gen_output.trajectories,
                key=lambda t: t.probability,
                reverse=True,
            )

        print("  Top 3 trajectories:")
        for i, traj in enumerate(sorted_trajs[:3]):
            text_preview = traj.text[:50].replace("\n", "\\n")
            if use_log_probs:
                print(
                    f"    {i + 1}. logp={traj.log_probability:.1f}: {text_preview}..."
                )
            else:
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
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


def run_visualization(output_dir: Path) -> None:
    """Run visualization on experiment results."""
    from visualize_experiment import visualize_results

    viz_dir = output_dir / "viz"
    print(f"\nGenerating visualizations in {viz_dir}")
    visualize_results(output_dir, viz_dir)


def build_prompts(params: Params) -> dict[str, str]:
    """Build prompt variants from params."""
    base_prompt = params.generation.base_prompt
    prompts = {"branch": base_prompt}

    for i, branch in enumerate(params.generation.branching_points):
        prompts[f"branch_{i}"] = base_prompt + branch

    return prompts
