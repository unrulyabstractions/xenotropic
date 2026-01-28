#!/usr/bin/env python3
"""
Run full experiment: collect trajectories, evaluate with judge, estimate cores.

This script:
1. Loads experiment parameters from a trial JSON file
2. Collects trajectories from the model using sampling
3. Evaluates each trajectory using judge-based systems
4. Computes cores and per-trajectory deviances
5. Outputs results and visualizations

Usage: python run_experiment.py [trial] [--no-viz]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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

from exploration import (
    CoreEstimator,
    CoreEstimatorConfig,
    ModelRunner,
    TrajectoryCollector,
    TrajectoryCollectorConfig,
)
from xenotechnics.structures.judge import JudgeStructure

# -----------------------------------------------------------------------------
# Trajectory Collection
# -----------------------------------------------------------------------------


def collect_trajectories(
    params: Params,
    model_runner: ModelRunner,
    prompt_variant: str,
    prompt_text: str,
    target_mass: float = 0.9,
    max_iterations: int = 200,
    verbose: bool = True,
) -> GenerationOutput:
    """Collect trajectories for a single prompt."""
    config = TrajectoryCollectorConfig(
        max_new_tokens=10,
        temperature=params.generation.temperature,
        top_k=params.generation.top_k,
        top_p=params.generation.top_p,
        target_mass=target_mass,
        max_iterations=max_iterations,
        max_trajectories=params.generation.max_trajectories,
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

    trajectory_records = [
        TrajectoryRecord(
            text=traj.text,
            probability=traj.probability,
            log_probability=traj.log_probability,
            per_token_logprobs=[
                {"token": tok, "logprob": lp}
                for tok, lp in zip(traj.tokens, traj.per_token_logprobs)
            ],
        )
        for traj in result.trajectories
    ]

    return GenerationOutput(
        param_id=params.param_id,
        experiment_id=params.experiment_id,
        prompt_variant=prompt_variant,
        prompt_text=prompt_text,
        model=params.generation.model,
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        total_mass=result.total_mass,
        num_trajectories=len(trajectory_records),
        trajectories=trajectory_records,
    )


# -----------------------------------------------------------------------------
# Core Estimation
# -----------------------------------------------------------------------------


def estimate_cores(
    params: Params,
    gen_output: GenerationOutput,
    judge_runner: ModelRunner,
    verbose: bool = True,
) -> CoreEstimationOutput:
    """Estimate cores for collected trajectories."""
    structures = params.estimation.structures
    system_names = params.estimation.systems

    if verbose:
        print(f"\n  Estimating cores for: {gen_output.prompt_variant}")
        print(f"  Trajectories: {gen_output.num_trajectories}")
        print(f"  Structures: {structures}")

    estimator = CoreEstimator(CoreEstimatorConfig(use_log_space=True))

    def make_scorer(structure: str):
        judge = JudgeStructure(question=structure, model_runner=judge_runner)
        return lambda text: judge.judge(text)[0]

    system_results = []
    for system_name in system_names:
        if verbose:
            print(f"\n  System: {system_name}")

        result = estimator.estimate(
            trajectories=gen_output.trajectories,
            structures=structures,
            scorer_factory=make_scorer,
            context_prefix=gen_output.prompt_text,
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
        judge_model=params.estimation.model,
        prompt_variant=gen_output.prompt_variant,
        prompt_text=gen_output.prompt_text,
        num_trajectories=gen_output.num_trajectories,
        total_mass=gen_output.total_mass,
        systems=system_results,
    )


# -----------------------------------------------------------------------------
# Main Experiment
# -----------------------------------------------------------------------------


def run_experiment(
    params: Params,
    target_mass: float = 0.9,
    max_iterations: int = 200,
    verbose: bool = True,
) -> tuple[list[GenerationOutput], list[CoreEstimationOutput]]:
    """Run full experiment: collect trajectories and estimate cores."""
    print("=" * 70)
    print(f"EXPERIMENT: {params.experiment_id}")
    print("=" * 70)
    print(f"Param ID: {params.param_id}")
    print(f"Generation model: {params.generation.model}")
    print(f"Judge model: {params.estimation.model}")
    print()

    prompts = build_prompts(params)
    print(f"Prompts ({len(prompts)}):")
    for name, prompt in prompts.items():
        print(f"  {name}: ...{prompt[-50:]}")
    print()

    # Load models
    print("Loading generation model...")
    gen_runner = ModelRunner(params.generation.model)
    print(f"  Device: {gen_runner.device}")

    if params.estimation.model == params.generation.model:
        print("Using same model for judging")
        judge_runner = gen_runner
    else:
        print("Loading judge model...")
        judge_runner = ModelRunner(params.estimation.model)
        print(f"  Device: {judge_runner.device}")
    print()

    # Collect and estimate
    gen_outputs = []
    for prompt_name, prompt_text in prompts.items():
        gen_output = collect_trajectories(
            params,
            gen_runner,
            prompt_name,
            prompt_text,
            target_mass,
            max_iterations,
            verbose,
        )
        gen_outputs.append(gen_output)

    est_outputs = []
    for gen_output in gen_outputs:
        est_output = estimate_cores(params, gen_output, judge_runner, verbose)
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

    output_dir = Path(__file__).parent / "out" / args.trial
    clean_output_dir(output_dir)
    print(f"Output directory: {output_dir}")

    gen_outputs, est_outputs = run_experiment(
        params,
        target_mass=args.target_mass,
        max_iterations=args.max_iterations,
        verbose=not args.quiet,
    )

    save_outputs(output_dir, gen_outputs, est_outputs)
    print_summary(gen_outputs, est_outputs, use_log_probs=True)

    if not args.no_viz:
        run_visualization(output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
