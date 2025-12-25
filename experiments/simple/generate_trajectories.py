"""
Generate trajectories using SamplingGenerator.

Iteratively samples from base_prompt and each base_prompt + branch_point,
saving trajectories after each iteration for incremental progress.
Reuses tree across iterations to avoid double-counting.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataclasses import asdict

from schemas import (
    EstimationConfig,
    GenerationConfig,
    GenerationOutput,
    Params,
    TrajectoryRecord,
)

from exploration import SamplingGenerator
from xenotechnics.common import String
from xenotechnics.trees.tree import TreeNode

# -----------------------------------------------------------------------------
# Input/Output Data Structures
# -----------------------------------------------------------------------------


@dataclass
class GenerateTrajectoriesInput:
    """Input for trajectory generation."""

    params_path: Path
    output_dir: Path
    min_prob_mass_per_iter: float
    target_total_mass: float
    max_iterations: int
    verbose: bool


@dataclass
class GenerateTrajectoriesOutput:
    """Output from trajectory generation."""

    experiment_id: str
    param_id: str
    variant_masses: dict[str, float]
    total_mass: float
    iterations_run: int


# -----------------------------------------------------------------------------
# Core Logic
# -----------------------------------------------------------------------------


def load_params(params_path: Path) -> Params:
    """Load experiment parameters from JSON file."""
    with open(params_path) as f:
        data = json.load(f)

    gen_config = GenerationConfig(**data["generation"])
    est_config = EstimationConfig(**data["estimation"])
    return Params(
        experiment_id=data["experiment_id"],
        generation=gen_config,
        estimation=est_config,
    )


def save_generation_output(output: GenerationOutput, output_dir: Path) -> Path:
    """Save generation output to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{output.param_id}_gen_{output.prompt_variant}_{output.timestamp}.json"
    filepath = output_dir / filename
    with open(filepath, "w") as f:
        json.dump(asdict(output), f, indent=2)
    print(f"Saved: {filepath}")
    return filepath


def get_prompt_variants(params: Params) -> list[tuple[str, str]]:
    """Get list of (variant_name, prompt_text) tuples."""
    gen_config = params.generation
    base_prompt = gen_config.base_prompt
    branching_points = gen_config.branching_points

    variants = [("base", base_prompt)]
    for branch in branching_points:
        variants.append((branch, base_prompt + branch))

    return variants


def generate_trajectories(inp: GenerateTrajectoriesInput) -> GenerateTrajectoriesOutput:
    """Main trajectory generation logic."""
    params = load_params(inp.params_path)
    gen_config = params.generation

    experiment_output_dir = inp.output_dir / params.output_dir_name
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"TRAJECTORY GENERATION: {params.experiment_id}")
    print(f"Param ID: {params.param_id}")
    print(f"Output dir: {experiment_output_dir}")
    print("=" * 80)
    print()
    print(f"Generator model: {gen_config.model}")
    print(f"Min prob mass per iteration: {inp.min_prob_mass_per_iter}")
    print(f"Target total mass: {inp.target_total_mass}")
    print(f"Max iterations: {inp.max_iterations}")
    print()

    variants = get_prompt_variants(params)
    print("Prompt variants:")
    for name, text in variants:
        print(f"  {name}: ...{text[-50:]}")
    print()

    temperature = gen_config.temperature
    top_p = gen_config.top_p
    top_k = gen_config.top_k
    seed = gen_config.seed

    print(f"Initializing SamplingGenerator with model: {gen_config.model}")
    generator = SamplingGenerator(model_name=gen_config.model)
    print()

    variant_trees: dict[str, TreeNode | None] = {name: None for name, _ in variants}
    variant_seen: dict[str, set[tuple[str, ...]]] = {
        name: set() for name, _ in variants
    }
    variant_masses: dict[str, float] = {name: 0.0 for name, _ in variants}

    iteration = 0
    no_progress_count = 0

    while True:
        for variant_name, prompt_text in variants:
            total_mass = sum(variant_masses.values())
            if total_mass >= inp.target_total_mass:
                print()
                print("=" * 80)
                print(f"TARGET REACHED: Total mass = {total_mass:.4f}")
                print("=" * 80)
                return GenerateTrajectoriesOutput(
                    experiment_id=params.experiment_id,
                    param_id=params.param_id,
                    variant_masses=variant_masses,
                    total_mass=total_mass,
                    iterations_run=iteration,
                )

            if iteration >= inp.max_iterations:
                print()
                print("=" * 80)
                print(
                    f"MAX ITERATIONS REACHED: {iteration} iterations, mass = {total_mass:.4f}"
                )
                print("=" * 80)
                return GenerateTrajectoriesOutput(
                    experiment_id=params.experiment_id,
                    param_id=params.param_id,
                    variant_masses=variant_masses,
                    total_mass=total_mass,
                    iterations_run=iteration,
                )

            if no_progress_count >= 20:
                print()
                print("=" * 80)
                print(
                    f"NO PROGRESS: {no_progress_count} iterations without new trajectories"
                )
                print(f"Final mass = {total_mass:.4f}")
                print("=" * 80)
                return GenerateTrajectoriesOutput(
                    experiment_id=params.experiment_id,
                    param_id=params.param_id,
                    variant_masses=variant_masses,
                    total_mass=total_mass,
                    iterations_run=iteration,
                )

            iteration += 1
            prev_mass = variant_masses[variant_name]

            print("=" * 80)
            print(f"ITERATION {iteration}: {variant_name}")
            print(
                f"Variant mass: {prev_mass:.4f}, Total mass: {total_mass:.4f} / {inp.target_total_mass}"
            )
            print("=" * 80)
            print()

            prompt = String.from_text(prompt_text)
            existing_tree = variant_trees[variant_name]

            if inp.verbose:
                print(f"  Searching for {inp.min_prob_mass_per_iter:.2f} more mass...")

            tree = generator.run(
                prompt=prompt,
                max_new_tokens=100,
                verbose=False,
                existing_tree=existing_tree,
                seed=seed + iteration,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

            variant_trees[variant_name] = tree

            trajectory_nodes = tree.get_trajectory_nodes()
            prompt_token_count = generator.prompt_token_count

            current_mass = sum(
                t.get_continuation_prob(prompt_token_count) for t in trajectory_nodes
            )
            variant_masses[variant_name] = current_mass

            new_trajectories = []
            seen = variant_seen[variant_name]
            for traj_node in trajectory_nodes:
                token_tuple = traj_node.string.tokens
                if token_tuple not in seen:
                    seen.add(token_tuple)
                    prob = traj_node.get_continuation_prob(prompt_token_count)
                    new_trajectories.append((traj_node, prob))

            if len(new_trajectories) > 0:
                no_progress_count = 0
            else:
                no_progress_count += 1

            if inp.verbose:
                mass_gained = current_mass - prev_mass
                print(
                    f"  Found {len(new_trajectories)} new trajectories, mass gained: {mass_gained:.4f}"
                )
                print(
                    f"  Total trajectories: {len(trajectory_nodes)}, Total mass: {current_mass:.4f}"
                )

            if new_trajectories:
                records = []
                for traj_node, prob in new_trajectories:
                    record = TrajectoryRecord(
                        text=traj_node.string.to_text(),
                        tokens=list(traj_node.string.tokens),
                        probability=float(prob),
                        log_probability=float(traj_node.path_logprob()),
                    )
                    records.append(record)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output = GenerationOutput(
                    param_id=params.param_id,
                    experiment_id=params.experiment_id,
                    prompt_variant=variant_name,
                    prompt_text=prompt_text,
                    model=gen_config.model,
                    timestamp=timestamp,
                    min_prob_mass=inp.min_prob_mass_per_iter,
                    total_mass=current_mass,
                    num_trajectories=len(records),
                    trajectories=records,
                )
                save_generation_output(output, experiment_output_dir)

                if inp.verbose:
                    print()
                    print("Top 3 new trajectories:")
                    sorted_trajs = sorted(
                        records, key=lambda t: t.probability, reverse=True
                    )[:3]
                    for rank, traj in enumerate(sorted_trajs, 1):
                        full_text = traj.text
                        if "assistant\n" in full_text:
                            continuation = (
                                full_text.split("assistant\n")[-1]
                                .replace("<|im_end|>", "")
                                .strip()
                            )
                        else:
                            continuation = full_text[-40:]
                        print(f"  {rank}. [p={traj.probability:.4f}] {continuation}")

            print()

            if current_mass - prev_mass < inp.min_prob_mass_per_iter * 0.1:
                if inp.verbose:
                    print("  Warning: Low mass gain, variant may be saturated")

    return GenerateTrajectoriesOutput(
        experiment_id=params.experiment_id,
        param_id=params.param_id,
        variant_masses=variant_masses,
        total_mass=sum(variant_masses.values()),
        iterations_run=iteration,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate trajectories with iterative sampling"
    )
    parser.add_argument("--trial", type=str, default="default", help="Trial name")
    parser.add_argument(
        "--params", type=Path, default=None, help="Direct path to params JSON"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent.parent / "outputs" / "simple",
    )
    parser.add_argument(
        "--min-mass",
        type=float,
        default=0.01,
        help="Minimum probability mass per iteration",
    )
    parser.add_argument(
        "--target-mass", type=float, default=0.95, help="Target total probability mass"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=500, help="Maximum iterations"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    return parser.parse_args()


def input_from_args(args: argparse.Namespace) -> GenerateTrajectoriesInput:
    """Load input from command line arguments."""
    if args.params is not None:
        params_path = args.params
    else:
        params_path = Path(__file__).parent / "trials" / f"{args.trial}.json"

    if not params_path.exists():
        raise FileNotFoundError(f"Params file not found: {params_path}")

    return GenerateTrajectoriesInput(
        params_path=params_path,
        output_dir=args.output,
        min_prob_mass_per_iter=args.min_mass,
        target_total_mass=args.target_mass,
        max_iterations=args.max_iterations,
        verbose=not args.quiet,
    )


def save_output(args: argparse.Namespace, output: GenerateTrajectoriesOutput) -> None:
    """Save output (saves during generation, summary here)."""
    pass


def print_output(args: argparse.Namespace, output: GenerateTrajectoriesOutput) -> None:
    """Print output summary."""
    print()
    print("=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print()
    print("Mass per variant:")
    for name, mass in output.variant_masses.items():
        print(f"  {name}: {mass:.4f}")
    print(f"Total: {output.total_mass:.4f}")
    print(f"Iterations: {output.iterations_run}")


def main() -> int:
    args = get_args()
    inp: GenerateTrajectoriesInput = input_from_args(args)
    output: GenerateTrajectoriesOutput = generate_trajectories(inp)

    save_output(args, output)
    print_output(args, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
