"""
Estimate cores from saved trajectories using Judge-based systems.

Loads all trajectory files for an experiment and computes system cores,
deviations, and homogenization metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from glob import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from schemas import CoreEstimationOutput, GenerationOutput, TrajectoryRecord

from xenotechnics.common import AbstractSystem, String
from xenotechnics.systems import JudgeVectorSystem
from xenotechnics.systems.judge_entropic_system import JudgeEntropicSystem
from xenotechnics.systems.judge_generalized_system import JudgeGeneralizedSystem

# -----------------------------------------------------------------------------
# Input/Output Data Structures
# -----------------------------------------------------------------------------


@dataclass
class EstimateCoresInput:
    """Input for core estimation."""

    params_path: Path
    output_dir: Path
    verbose: bool


@dataclass
class EstimateCoresOutput:
    """Output from core estimation."""

    estimation_output: CoreEstimationOutput
    system_names: list[str]
    output_filepath: Path


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def load_experiment_params(params_path: Path) -> dict:
    """Load experiment parameters from JSON file."""
    with open(params_path) as f:
        return json.load(f)


def load_generation_files(
    output_dir: Path, experiment_id: str
) -> dict[str, list[GenerationOutput]]:
    """
    Load all generation files for an experiment.

    Returns dict mapping prompt_variant -> list of GenerationOutput.
    """
    pattern = str(output_dir / f"{experiment_id}_gen_*.json")
    files = glob(pattern)

    if not files:
        raise FileNotFoundError(f"No generation files found matching: {pattern}")

    print(f"Found {len(files)} generation files")

    # Group by prompt variant
    outputs_by_variant: dict[str, list[GenerationOutput]] = {}

    for filepath in sorted(files):
        with open(filepath) as f:
            data = json.load(f)

        # Reconstruct GenerationOutput
        trajectories = [
            TrajectoryRecord(
                text=t["text"],
                tokens=t["tokens"],
                probability=t["probability"],
                log_probability=t["log_probability"],
            )
            for t in data["trajectories"]
        ]

        output = GenerationOutput(
            param_id=data["param_id"],
            experiment_id=data["experiment_id"],
            prompt_variant=data["prompt_variant"],
            prompt_text=data["prompt_text"],
            model=data["model"],
            timestamp=data["timestamp"],
            min_prob_mass=data["min_prob_mass"],
            total_mass=data["total_mass"],
            num_trajectories=data["num_trajectories"],
            trajectories=trajectories,
        )

        variant = output.prompt_variant
        if variant not in outputs_by_variant:
            outputs_by_variant[variant] = []
        outputs_by_variant[variant].append(output)

    return outputs_by_variant


def merge_trajectories(
    outputs: list[GenerationOutput],
) -> tuple[list[String], np.ndarray, float]:
    """
    Merge trajectories from multiple generation outputs, deduplicating by tokens.

    Returns (trajectory_strings, probabilities, total_mass).
    """
    seen_tokens: set[tuple[int, ...]] = set()
    all_strings = []
    all_probs = []
    total_mass = 0.0

    for output in outputs:
        for traj in output.trajectories:
            token_tuple = tuple(traj.tokens)
            if token_tuple in seen_tokens:
                continue
            seen_tokens.add(token_tuple)
            string = String(tokens=token_tuple)
            all_strings.append(string)
            all_probs.append(traj.probability)
        total_mass += output.total_mass

    return all_strings, np.array(all_probs), total_mass


def create_systems(params: dict) -> tuple[list[AbstractSystem], list[str]]:
    """
    Create evaluation systems based on experiment params.

    Returns (systems, system_names).
    """
    est_config = params["estimation"]
    model = est_config["model"]
    structures = est_config["structures"]
    system_specs = est_config["systems"]

    systems = []
    names = []

    for spec in system_specs:
        if spec == "vector_system":
            system = JudgeVectorSystem(questions=structures, model_name=model)
            names.append("VectorSystem (L2^2)")
        elif spec.startswith("generalized_system"):
            # Parse r and q from spec like "generalized_system_q1_r1"
            parts = spec.split("_")
            q = float(parts[-1].replace("r", ""))
            r = float(parts[-2].replace("q", ""))
            system = JudgeGeneralizedSystem(
                questions=structures, model_name=model, r=r, q=q
            )
            names.append(f"GeneralizedSystem (r={r},q={q})")
        elif spec.startswith("entropic_system"):
            # Parse q from spec like "entropic_system_q1"
            parts = spec.split("_")
            q = float(parts[-1].replace("q", ""))
            system = JudgeEntropicSystem(
                questions=structures, model_name=model, q=q, mode="excess"
            )
            names.append(f"EntropicSystem (q={q}, excess)")
        else:
            raise ValueError(f"Unknown system spec: {spec}")

        systems.append(system)

    return systems, names


def compute_system_statistics(
    trajectory_strings: list[String],
    probabilities: np.ndarray,
    systems: list[AbstractSystem],
    verbose: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Compute cores and deviations for all systems.

    Returns (cores, deviations) lists.
    """
    cores = []
    all_deviations = []

    for sys_idx, system in enumerate(systems):
        if verbose:
            print(f"  Computing statistics for system {sys_idx + 1}/{len(systems)}...")

        # Compute core
        core = system.compute_core(trajectory_strings, probabilities)
        cores.append(core.to_array())

        # Compute deviations
        deviations = []
        for traj in trajectory_strings:
            compliance = system.compliance(traj)
            orientation = compliance.to_array() - core.to_array()
            # Create temporary compliance for orientation
            orientation_compliance = type(compliance)(
                system=system, compliance_vector=orientation, string=None
            )
            deviation = system.score_operator(orientation_compliance)
            deviations.append(deviation)

        all_deviations.append(np.array(deviations))

    return cores, all_deviations


# -----------------------------------------------------------------------------
# Core Logic
# -----------------------------------------------------------------------------


def estimate_cores(inp: EstimateCoresInput) -> EstimateCoresOutput:
    """Main core estimation logic."""
    params = load_experiment_params(inp.params_path)
    experiment_id = params["experiment_id"]
    est_config = params["estimation"]

    print("=" * 80)
    print(f"CORE ESTIMATION: Experiment {experiment_id}")
    print("=" * 80)
    print()
    print(f"Judge model: {est_config['model']}")
    print()

    # Load all generation files
    print("Loading trajectory files...")
    outputs_by_variant = load_generation_files(inp.output_dir, experiment_id)
    print()

    for variant, outputs in outputs_by_variant.items():
        total_trajs = sum(o.num_trajectories for o in outputs)
        total_mass = sum(o.total_mass for o in outputs)
        print(
            f"  {variant}: {len(outputs)} files, {total_trajs} trajectories, mass={total_mass:.4f}"
        )
    print()

    # Create systems
    print("Creating evaluation systems...")
    systems, system_names = create_systems(params)
    for i, name in enumerate(system_names, 1):
        print(f"  {i}. {name}")
    print()

    # Results storage
    cores_by_variant: dict[str, list[list[float]]] = {}
    expected_dev_by_variant: dict[str, list[float]] = {}
    var_dev_by_variant: dict[str, list[float]] = {}
    num_trajs_by_variant: dict[str, int] = {}
    mass_by_variant: dict[str, float] = {}

    # Process each variant
    for variant, outputs in outputs_by_variant.items():
        print("=" * 80)
        print(f"COMPUTING CORES: {variant}")
        print("=" * 80)
        print()

        # Merge trajectories
        trajectory_strings, probabilities, total_mass = merge_trajectories(outputs)
        num_trajs_by_variant[variant] = len(trajectory_strings)
        mass_by_variant[variant] = total_mass

        print(
            f"  Merged {len(trajectory_strings)} trajectories, total mass={total_mass:.4f}"
        )
        print()

        # Compute statistics
        cores, deviations = compute_system_statistics(
            trajectory_strings, probabilities, systems, verbose=inp.verbose
        )

        # Store results
        cores_by_variant[variant] = [c.tolist() for c in cores]

        # Compute expected and variance of deviance
        exp_devs = []
        var_devs = []
        for dev_array in deviations:
            exp_dev = float(np.average(dev_array, weights=probabilities))
            var_dev = float(
                np.average((dev_array - exp_dev) ** 2, weights=probabilities)
            )
            exp_devs.append(exp_dev)
            var_devs.append(var_dev)

        expected_dev_by_variant[variant] = exp_devs
        var_dev_by_variant[variant] = var_devs
        print()

    # Build output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    estimation_output = CoreEstimationOutput(
        experiment_id=experiment_id,
        timestamp=timestamp,
        judge_model=est_config["model"],
        structures=est_config["structures"],
        systems=est_config["systems"],
        prompt_variants=list(outputs_by_variant.keys()),
        cores=cores_by_variant,
        expected_deviance=expected_dev_by_variant,
        var_deviance=var_dev_by_variant,
        num_trajectories=num_trajs_by_variant,
        total_mass=mass_by_variant,
    )

    filename = f"{experiment_id}_est_{timestamp}.json"
    filepath = inp.output_dir / filename

    return EstimateCoresOutput(
        estimation_output=estimation_output,
        system_names=system_names,
        output_filepath=filepath,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Estimate cores from trajectory files")
    parser.add_argument(
        "--params",
        type=Path,
        default=Path(__file__).parent / "experiment_params.json",
        help="Path to experiment parameters JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent.parent / "outputs" / "simple",
        help="Directory containing trajectory files (and where to save results)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    return parser.parse_args()


def input_from_args(args: argparse.Namespace) -> EstimateCoresInput:
    """Load input from command line arguments."""
    return EstimateCoresInput(
        params_path=args.params,
        output_dir=args.output,
        verbose=not args.quiet,
    )


def save_output(args: argparse.Namespace, output: EstimateCoresOutput) -> None:
    """Save estimation output to JSON file."""
    output.output_filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(output.output_filepath, "w") as f:
        json.dump(asdict(output.estimation_output), f, indent=2)
    print(f"Saved results to: {output.output_filepath}")
    print()


def print_output(args: argparse.Namespace, output: EstimateCoresOutput) -> None:
    """Print results summary."""
    est = output.estimation_output
    system_names = output.system_names

    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()

    for sys_idx, system_name in enumerate(system_names):
        print(f"System: {system_name}")
        print("-" * 80)
        print()

        for variant in est.prompt_variants:
            core = est.cores[variant][sys_idx]
            exp_dev = est.expected_deviance[variant][sys_idx]
            var_dev = est.var_deviance[variant][sys_idx]

            print(f"  {variant}:")
            if len(core) == 3:
                print(
                    f"    Core <Lambda> = [{core[0]:.3f}, {core[1]:.3f}, {core[2]:.3f}]"
                )
                print("                    (queer, women, men)")
            else:
                print(f"    Core <Lambda> = {core}")
            print(f"    E[d] = {exp_dev:.4f}, Var[d] = {var_dev:.4f}")
            print()

        print()

    print("=" * 80)
    print("CORE ESTIMATION COMPLETE")
    print("=" * 80)
    print()


def main() -> int:
    args = get_args()
    inp: EstimateCoresInput = input_from_args(args)
    output: EstimateCoresOutput = estimate_cores(inp)

    save_output(args, output)
    print_output(args, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
