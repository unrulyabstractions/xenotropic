#!/usr/bin/env python3
"""
Run full experiment: collect trajectories, evaluate with judge, estimate cores.

This script:
1. Loads experiment parameters from a trial JSON file
2. Collects trajectories from the model using sampling
3. Evaluates each trajectory using judge-based systems
4. Computes cores and per-trajectory deviances
5. Outputs results in structured format
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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

from exploration import (
    CoreEstimator,
    CoreEstimatorConfig,
    ModelRunner,
    TrajectoryCollector,
    TrajectoryCollectorConfig,
)
from xenotechnics.structures.judge import JudgeStructure

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def create_system(
    system_name: str,
    structures: list[str],
    model_runner: ModelRunner,
) -> tuple:
    """
    Create a judge-based system for evaluation.

    Args:
        system_name: Name of system type (vector_system, generalized_system_q1_r1, etc.)
        structures: List of structure questions
        model_runner: ModelRunner for the judge

    Returns:
        Tuple of (system, system_display_name)
    """
    from xenotechnics.systems import (
        JudgeEntropicSystem,
        JudgeGeneralizedSystem,
        JudgeVectorSystem,
    )

    if system_name == "vector_system":
        system = JudgeVectorSystem(
            questions=structures,
            model_runner=model_runner,
        )
        return system, "VectorSystem (L2)"

    elif system_name.startswith("generalized_system"):
        # Parse q and r from name like "generalized_system_q1_r1"
        parts = system_name.split("_")
        q = 1.0
        r = 1.0
        for part in parts:
            if part.startswith("q"):
                q = float(part[1:])
            elif part.startswith("r"):
                r = float(part[1:])
        system = JudgeGeneralizedSystem(
            questions=structures,
            model_runner=model_runner,
            q=q,
            r=r,
        )
        return system, f"GeneralizedSystem (q={q}, r={r})"

    elif system_name.startswith("entropic_system"):
        # Parse q from name like "entropic_system_q1"
        parts = system_name.split("_")
        q = 1.0
        for part in parts:
            if part.startswith("q"):
                q = float(part[1:])
        system = JudgeEntropicSystem(
            questions=structures,
            model_runner=model_runner,
            q=q,
            mode="excess",
        )
        return system, f"EntropicSystem (q={q})"

    else:
        raise ValueError(f"Unknown system type: {system_name}")


def collect_trajectories(
    params: Params,
    model_runner: ModelRunner,
    prompt_variant: str,
    prompt_text: str,
    target_mass: float = 0.9,
    max_iterations: int = 200,
    verbose: bool = True,
) -> GenerationOutput:
    """
    Collect trajectories for a single prompt.

    Args:
        params: Experiment parameters
        model_runner: ModelRunner for generation
        prompt_variant: Name of this prompt variant
        prompt_text: The actual prompt text
        target_mass: Target probability mass to collect
        max_iterations: Maximum collection iterations
        verbose: Print progress

    Returns:
        GenerationOutput with collected trajectories
    """
    config = TrajectoryCollectorConfig(
        max_new_tokens=10,  # Very short completions for better probability coverage
        temperature=params.generation.temperature,
        top_k=params.generation.top_k,
        top_p=params.generation.top_p,
        target_mass=target_mass,
        max_iterations=max_iterations,
        seed=params.generation.seed,
    )

    collector = TrajectoryCollector(model_runner, config)

    if verbose:
        print(f"\n  Collecting trajectories for: {prompt_variant}")
        print(f"  Prompt: {prompt_text[:60]}...")

    def progress_callback(progress):
        if verbose and progress.iteration % 10 == 0:
            print(
                f"    Iter {progress.iteration}: "
                f"{progress.trajectories_found} trajectories, "
                f"mass={progress.total_mass:.4f}"
            )

    result = collector.collect(prompt_text, progress_callback=progress_callback)

    if verbose:
        print(f"  Done: {len(result.trajectories)} unique trajectories")

    # Convert to TrajectoryRecord format
    trajectory_records = []
    for traj in result.trajectories:
        record = TrajectoryRecord(
            text=traj.text,
            probability=traj.probability,
            log_probability=traj.log_probability,
            per_token_logprobs=[
                {"token": tok, "logprob": lp}
                for tok, lp in zip(traj.tokens, traj.per_token_logprobs)
            ],
        )
        trajectory_records.append(record)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return GenerationOutput(
        param_id=params.param_id,
        experiment_id=params.experiment_id,
        prompt_variant=prompt_variant,
        prompt_text=prompt_text,
        model=params.generation.model,
        timestamp=timestamp,
        total_mass=result.total_mass,
        num_trajectories=len(trajectory_records),
        trajectories=trajectory_records,
    )


def estimate_cores(
    params: Params,
    gen_output: GenerationOutput,
    judge_runner: ModelRunner,
    verbose: bool = True,
) -> CoreEstimationOutput:
    """
    Estimate cores for collected trajectories using CoreEstimator.

    Args:
        params: Experiment parameters
        gen_output: Generation output with trajectories
        judge_runner: ModelRunner for judge evaluation
        verbose: Print progress

    Returns:
        CoreEstimationOutput with core estimates
    """
    structures = params.estimation.structures
    system_names = params.estimation.systems

    if verbose:
        print(f"\n  Estimating cores for: {gen_output.prompt_variant}")
        print(f"  Trajectories: {gen_output.num_trajectories}")
        print(f"  Structures: {structures}")

    # Create CoreEstimator
    estimator = CoreEstimator(CoreEstimatorConfig(use_log_space=True))

    # Create scorer factory that wraps JudgeStructure
    def make_scorer(structure: str):
        judge = JudgeStructure(question=structure, model_runner=judge_runner)

        def scorer(text: str) -> float:
            score, _ = judge.judge(text)
            return score

        return scorer

    system_results = []

    for system_name in system_names:
        _, display_name = create_system(system_name, structures, judge_runner)

        if verbose:
            print(f"\n  System: {display_name}")

        # Use CoreEstimator to compute cores
        result = estimator.estimate(
            trajectories=gen_output.trajectories,
            structures=structures,
            scorer_factory=make_scorer,
            context_prefix=gen_output.prompt_text,
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
        judge_model=params.estimation.model,
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
    output_dir: Path,
    target_mass: float = 0.9,
    max_iterations: int = 200,
    verbose: bool = True,
) -> tuple[list[GenerationOutput], list[CoreEstimationOutput]]:
    """
    Run full experiment: collect trajectories and estimate cores.

    Args:
        params: Experiment parameters
        output_dir: Directory for output files
        target_mass: Target probability mass to collect
        max_iterations: Maximum collection iterations
        verbose: Print progress

    Returns:
        Tuple of (generation_outputs, estimation_outputs)
    """
    print("=" * 70)
    print(f"EXPERIMENT: {params.experiment_id}")
    print("=" * 70)
    print(f"Param ID: {params.param_id}")
    print(f"Generation model: {params.generation.model}")
    print(f"Judge model: {params.estimation.model}")
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

    # Load generation model
    print("Loading generation model...")
    gen_runner = ModelRunner(params.generation.model)
    print(f"  Device: {gen_runner.device}")
    print()

    # Load judge model (may be same as generation model)
    if params.estimation.model == params.generation.model:
        print("Using same model for judging")
        judge_runner = gen_runner
    else:
        print("Loading judge model...")
        judge_runner = ModelRunner(params.estimation.model)
        print(f"  Device: {judge_runner.device}")
    print()

    # Collect trajectories for each prompt
    gen_outputs = []
    for prompt_name, prompt_text in prompts.items():
        gen_output = collect_trajectories(
            params=params,
            model_runner=gen_runner,
            prompt_variant=prompt_name,
            prompt_text=prompt_text,
            target_mass=target_mass,
            max_iterations=max_iterations,
            verbose=verbose,
        )
        gen_outputs.append(gen_output)

    # Estimate cores for each prompt
    est_outputs = []
    for gen_output in gen_outputs:
        est_output = estimate_cores(
            params=params,
            gen_output=gen_output,
            judge_runner=judge_runner,
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
        default="test",
        help="Trial name (without .json) from trials/ directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent.parent / "outputs" / "simple",
        help="Base output directory",
    )
    parser.add_argument(
        "--target-mass",
        type=float,
        default=0.9,
        help="Target probability mass to collect",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=200,
        help="Maximum collection iterations",
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


def clean_output_dir(output_dir: Path) -> None:
    """Remove all files in output directory."""
    if output_dir.exists():
        import shutil

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
) -> None:
    """Print summary of results."""
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for gen_output, est_output in zip(gen_outputs, est_outputs):
        print(f"\nPrompt: {gen_output.prompt_variant}")
        print(f"  Trajectories: {gen_output.num_trajectories}")

        # Show top trajectories (use log probs for sorting)
        sorted_trajs = sorted(
            gen_output.trajectories,
            key=lambda t: t.log_probability,
            reverse=True,
        )
        print("  Top 3 trajectories:")
        for i, traj in enumerate(sorted_trajs[:3]):
            text_preview = traj.text[:50].replace("\n", "\\n")
            print(f"    {i + 1}. logp={traj.log_probability:.1f}: {text_preview}...")

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


def main() -> int:
    args = get_args()

    # Load parameters
    params = load_params(args.trial)

    # Output directory: experiments/simple/out/{trial_name}/
    output_dir = Path(__file__).parent / "out" / args.trial

    # Clean previous results
    clean_output_dir(output_dir)
    print(f"Output directory: {output_dir}")

    # Run experiment
    gen_outputs, est_outputs = run_experiment(
        params=params,
        output_dir=output_dir,
        target_mass=args.target_mass,
        max_iterations=args.max_iterations,
        verbose=not args.quiet,
    )

    # Save results
    save_outputs(output_dir, gen_outputs, est_outputs)

    # Print summary
    print_summary(gen_outputs, est_outputs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
