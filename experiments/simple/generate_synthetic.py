"""
Generate synthetic test data for visualization development.

Creates fake generation and estimation outputs that can be used to
test and iterate on visualization without running full experiments.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
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
# Input/Output Data Structures
# -----------------------------------------------------------------------------


@dataclass
class GenerateSyntheticInput:
    """Input for synthetic data generation."""

    params: Params
    output_dir: Path
    num_trajectories: int
    seed: int


@dataclass
class GenerateSyntheticOutput:
    """Output from synthetic data generation."""

    gen_outputs: list[GenerationOutput]
    est_outputs: list[CoreEstimationOutput]
    gen_filepaths: list[Path]
    est_filepaths: list[Path]


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def create_synthetic_trajectories(
    params: Params,
    num_trajectories: int,
    seed: int,
) -> list[GenerationOutput]:
    """Generate synthetic trajectory data."""
    random.seed(seed)
    np.random.seed(seed)

    # Sample continuations with varying probabilities
    continuations = [
        ("Beautiful.", 0.25),
        ("beautiful.", 0.11),
        ("red.", 0.08),
        ("Pretty.", 0.06),
        ("delicate.", 0.05),
        ("lovely.", 0.04),
        ("amazing.", 0.03),
        ("perfect.", 0.03),
        ("wonderful.", 0.02),
        ("stunning.", 0.02),
        ("gorgeous.", 0.02),
        ("sweet.", 0.02),
        ("nice.", 0.01),
        ("fine.", 0.01),
        ("great.", 0.01),
        ("blooming.", 0.01),
        ("thorny.", 0.005),
        ("fragrant.", 0.005),
        ("wilting.", 0.002),
        ("dying.", 0.001),
    ]

    # Normalize probabilities
    total_prob = sum(p for _, p in continuations)
    continuations = [(c, p / total_prob) for c, p in continuations]

    base_prompt = params.generation.base_prompt
    prompt_text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{base_prompt}<|im_end|>\n<|im_start|>assistant\n"

    trajectories = []
    total_mass = 0.0

    for i, (continuation, prob) in enumerate(continuations[:num_trajectories]):
        full_text = prompt_text + continuation + "<|im_end|>"
        tokens = list(full_text)  # Simplified tokenization

        traj = TrajectoryRecord(
            text=full_text,
            tokens=tokens,
            probability=prob,
            log_probability=float(np.log(prob + 1e-10)),
        )
        trajectories.append(traj)
        total_mass += prob

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = GenerationOutput(
        param_id=params.param_id,
        experiment_id=params.experiment_id,
        prompt_variant="base",
        prompt_text=base_prompt,
        model=params.generation.model,
        timestamp=timestamp,
        min_prob_mass=0.01,
        total_mass=total_mass,
        num_trajectories=len(trajectories),
        trajectories=trajectories,
    )

    return [output]


def create_synthetic_estimation(
    params: Params,
    gen_outputs: list[GenerationOutput],
    seed: int,
) -> list[CoreEstimationOutput]:
    """Generate synthetic core estimation data."""
    random.seed(seed)
    np.random.seed(seed)

    outputs = []

    for gen_output in gen_outputs:
        num_trajs = gen_output.num_trajectories
        structures = params.estimation.structures
        systems = params.estimation.systems

        system_results = []

        for system in systems:
            structure_results = []

            for structure in structures:
                # Generate random scores per trajectory
                # Make some trajectories score higher for certain structures
                scores = []
                for i, traj in enumerate(gen_output.trajectories):
                    # Base score depends on trajectory probability
                    base_score = 0.5 + 0.3 * np.random.randn()

                    # Add structure-specific bias
                    if "positive" in structure.lower():
                        # Positive words get higher scores
                        if any(
                            w in traj.text.lower()
                            for w in ["beautiful", "lovely", "amazing", "wonderful"]
                        ):
                            base_score += 0.3
                    elif "negative" in structure.lower():
                        # Negative words get higher scores
                        if any(
                            w in traj.text.lower()
                            for w in ["dying", "wilting", "thorny"]
                        ):
                            base_score += 0.3

                    scores.append(np.clip(base_score, 0, 1))

                # Compute core as probability-weighted average
                probs = np.array([t.probability for t in gen_output.trajectories])
                probs = probs / probs.sum()
                core = float(np.sum(np.array(scores) * probs))

                # Compute deviance
                deviances = np.abs(np.array(scores) - core)
                expected_deviance = float(np.sum(deviances * probs))
                var_deviance = float(
                    np.sum((deviances - expected_deviance) ** 2 * probs)
                )

                structure_results.append(
                    StructureResult(
                        structure=structure,
                        scores=scores,
                        core=core,
                        expected_deviance=expected_deviance,
                        var_deviance=var_deviance,
                    )
                )

            # Aggregate across structures
            cores = [s.core for s in structure_results]
            deviances = [s.expected_deviance for s in structure_results]

            system_results.append(
                SystemResult(
                    system=system,
                    structures=structure_results,
                    aggregate_core=float(np.mean(cores)),
                    aggregate_deviance=float(np.mean(deviances)),
                )
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        est_output = CoreEstimationOutput(
            param_id=params.param_id,
            experiment_id=params.experiment_id,
            timestamp=timestamp,
            judge_model=params.estimation.model,
            prompt_variant=gen_output.prompt_variant,
            prompt_text=gen_output.prompt_text,
            num_trajectories=num_trajs,
            total_mass=gen_output.total_mass,
            systems=system_results,
        )
        outputs.append(est_output)

    return outputs


# -----------------------------------------------------------------------------
# Core Logic
# -----------------------------------------------------------------------------


def generate_synthetic(inp: GenerateSyntheticInput) -> GenerateSyntheticOutput:
    """Main synthetic data generation logic."""
    params = inp.params

    print(f"Generating synthetic data for: {params.experiment_id}")
    print(f"Param ID: {params.param_id}")
    print(f"Output dir: {inp.output_dir}")
    print()

    # Generate trajectories
    gen_outputs = create_synthetic_trajectories(
        params, num_trajectories=inp.num_trajectories, seed=inp.seed
    )

    # Generate estimation
    est_outputs = create_synthetic_estimation(params, gen_outputs, seed=inp.seed)

    # Compute output filepaths
    gen_filepaths = []
    for gen_output in gen_outputs:
        filename = f"{gen_output.param_id}_gen_{gen_output.prompt_variant}_{gen_output.timestamp}.json"
        gen_filepaths.append(inp.output_dir / filename)

    est_filepaths = []
    for est_output in est_outputs:
        filename = f"{est_output.param_id}_est_{est_output.prompt_variant}_{est_output.timestamp}.json"
        est_filepaths.append(inp.output_dir / filename)

    return GenerateSyntheticOutput(
        gen_outputs=gen_outputs,
        est_outputs=est_outputs,
        gen_filepaths=gen_filepaths,
        est_filepaths=est_filepaths,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic test data")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent.parent / "outputs" / "simple",
        help="Base output directory",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=20,
        help="Number of trajectories to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def input_from_args(args: argparse.Namespace) -> GenerateSyntheticInput:
    """Load input from command line arguments."""
    # Load synthetic trial params
    params_path = Path(__file__).parent / "trials" / "synthetic.json"
    with open(params_path) as f:
        data = json.load(f)

    gen_config = GenerationConfig(**data["generation"])
    est_config = EstimationConfig(**data["estimation"])
    params = Params(
        experiment_id=data["experiment_id"],
        generation=gen_config,
        estimation=est_config,
    )

    output_dir = args.output / params.output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    return GenerateSyntheticInput(
        params=params,
        output_dir=output_dir,
        num_trajectories=args.num_trajectories,
        seed=args.seed,
    )


def save_output(args: argparse.Namespace, output: GenerateSyntheticOutput) -> None:
    """Save generation and estimation outputs to JSON files."""
    for gen_output, filepath in zip(output.gen_outputs, output.gen_filepaths):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(asdict(gen_output), f, indent=2)
        print(f"Saved: {filepath}")

    for est_output, filepath in zip(output.est_outputs, output.est_filepaths):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(asdict(est_output), f, indent=2)
        print(f"Saved: {filepath}")


def print_output(args: argparse.Namespace, output: GenerateSyntheticOutput) -> None:
    """Print summary of generated data."""
    print()
    print("Done! Generated:")
    print(f"  - {len(output.gen_outputs)} generation output(s)")
    print(f"  - {len(output.est_outputs)} estimation output(s)")


def main() -> int:
    args = get_args()
    inp: GenerateSyntheticInput = input_from_args(args)
    output: GenerateSyntheticOutput = generate_synthetic(inp)

    save_output(args, output)
    print_output(args, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
