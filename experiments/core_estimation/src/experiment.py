"""Experiment runner for trajectory collection and core estimation."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

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

BASE_DIR = Path(__file__).parent.parent  # core_estimation/


class Experiment:
    """Manages experiment lifecycle: run, save, visualize."""

    def __init__(self, params: Params, output_dir: Path):
        self.params = params
        self.output_dir = output_dir
        self.gen_outputs: list[GenerationOutput] = []
        self.est_outputs: list[CoreEstimationOutput] = []

    @property
    def is_synthetic(self) -> bool:
        return self.params.generation.model.lower() == "synthetic"

    @classmethod
    def from_trial(cls, trial_name: str) -> Experiment:
        """Load experiment from trial config file."""
        trial_path = BASE_DIR / "trials" / f"{trial_name}.json"
        if not trial_path.exists():
            available = [p.stem for p in (BASE_DIR / "trials").glob("*.json")]
            raise FileNotFoundError(
                f"Trial '{trial_name}' not found. Available: {available}"
            )

        with open(trial_path) as f:
            data = json.load(f)

        params = Params(
            experiment_id=data["experiment_id"],
            generation=GenerationConfig(**data["generation"]),
            estimation=EstimationConfig(**data["estimation"]),
        )

        return cls(params, BASE_DIR / "out" / trial_name)

    def run(self, verbose: bool = True, **kwargs) -> None:
        """Run the experiment."""
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._print_header(verbose)

        if self.is_synthetic:
            self._run_synthetic(verbose=verbose, **kwargs)
        else:
            self._run_real(verbose=verbose, **kwargs)

    def _print_header(self, verbose: bool):
        if not verbose:
            return
        p = self.params
        mode = "SYNTHETIC" if self.is_synthetic else ""
        print(f"Output: {self.output_dir}")
        print("=" * 60)
        print(f"EXPERIMENT {mode}: {p.experiment_id}")
        print("=" * 60)
        print(f"Model: {p.generation.model}")
        print()

    def _build_prompts(self) -> dict[str, str]:
        base = self.params.generation.base_prompt
        prompts = {"branch": base}
        for i, branch in enumerate(self.params.generation.branching_points):
            prompts[f"branch_{i}"] = base + branch
        return prompts

    # -------------------------------------------------------------------------
    # Real experiment
    # -------------------------------------------------------------------------

    def _run_real(
        self,
        target_mass: float = 0.9,
        max_iterations: int = 200,
        verbose: bool = True,
    ):
        from exploration import (
            ModelRunner,
            TrajectoryCollector,
            TrajectoryCollectorConfig,
        )
        from xenotechnics.structures.judge import JudgeStructure

        p = self.params
        prompts = self._build_prompts()

        if verbose:
            print("Loading model...")
        runner = ModelRunner(p.generation.model)

        # Collect trajectories
        for name, prompt in prompts.items():
            if verbose:
                print(f"\nCollecting: {name}")

            config = TrajectoryCollectorConfig(
                max_new_tokens=10,
                temperature=p.generation.temperature,
                top_k=p.generation.top_k,
                top_p=p.generation.top_p,
                target_mass=target_mass,
                max_iterations=max_iterations,
                max_trajectories=p.generation.max_trajectories,
                seed=p.generation.seed,
            )
            result = TrajectoryCollector(runner, config).collect(prompt)

            self._store_generation(
                name,
                prompt,
                result.formatted_prompt or prompt,
                result.trajectories,
                result.total_mass,
            )

            if verbose:
                print(
                    f"  {len(result.trajectories)} trajectories, mass={result.total_mass:.3f}"
                )

        # Load judge model if different
        judge_runner = runner
        if p.estimation.model != p.generation.model:
            if verbose:
                print(f"\nLoading judge model: {p.estimation.model}...")
            judge_runner = ModelRunner(p.estimation.model)

        # Estimate cores
        def make_scorer(struct: str) -> Callable[[str], float]:
            judge = JudgeStructure(question=struct, model_runner=judge_runner)
            return lambda text: judge.judge(text)[0]

        self._estimate_cores(make_scorer, verbose)

    def _store_generation(
        self,
        name: str,
        prompt: str,
        formatted_prompt: str,
        trajectories,
        total_mass: float,
    ):
        """Store generation output from collected trajectories."""
        p = self.params
        records = [
            TrajectoryRecord(
                text=t.text,
                probability=t.probability,
                log_probability=t.log_probability,
                per_token_logprobs=[
                    {"token": tok, "logprob": lp}
                    for tok, lp in zip(t.tokens, t.per_token_logprobs)
                ],
                is_greedy=getattr(t, "is_greedy", False),
            )
            for t in trajectories
        ]

        self.gen_outputs.append(
            GenerationOutput(
                param_id=p.param_id,
                experiment_id=p.experiment_id,
                prompt_variant=name,
                prompt_text=prompt,
                formatted_prompt=formatted_prompt,
                model=p.generation.model,
                timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                total_mass=total_mass,
                num_trajectories=len(records),
                trajectories=records,
            )
        )

    # -------------------------------------------------------------------------
    # Synthetic experiment
    # -------------------------------------------------------------------------

    def _run_synthetic(self, seed: int = 42, verbose: bool = True, **_):
        from synthetic import SyntheticGenerator, SyntheticScorer

        p = self.params
        prompts = self._build_prompts()
        num_traj = p.generation.max_trajectories or 20

        gen = SyntheticGenerator(seed)
        scorer = SyntheticScorer(seed + 1000)

        for name, prompt in prompts.items():
            if verbose:
                print(f"\nGenerating: {name}")

            trajectories, mass = gen.generate(prompt, num_traj)

            self.gen_outputs.append(
                GenerationOutput(
                    param_id=p.param_id,
                    experiment_id=p.experiment_id,
                    prompt_variant=name,
                    prompt_text=prompt,
                    formatted_prompt=prompt,  # No chat template for synthetic
                    model="synthetic",
                    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                    total_mass=mass,
                    num_trajectories=len(trajectories),
                    trajectories=trajectories,
                )
            )

            if verbose:
                print(f"  {len(trajectories)} trajectories, mass={mass:.3f}")

        def make_scorer(struct: str) -> Callable[[str], float]:
            return lambda text: scorer.score(text, struct)

        self._estimate_cores(make_scorer, verbose, use_log_space=False)

    # -------------------------------------------------------------------------
    # Core estimation (shared)
    # -------------------------------------------------------------------------

    def _estimate_cores(
        self,
        make_scorer: Callable[[str], Callable[[str], float]],
        verbose: bool,
        use_log_space: bool = True,
    ):
        from exploration import CoreEstimator, CoreEstimatorConfig

        p = self.params
        estimator = CoreEstimator(CoreEstimatorConfig(use_log_space=use_log_space))

        for gen in self.gen_outputs:
            if verbose:
                print(f"\nEstimating cores: {gen.prompt_variant}")

            result = estimator.estimate(
                gen.trajectories,
                p.estimation.structures,
                make_scorer,
                gen.prompt_text,
            )

            self.est_outputs.append(self._make_est_output(gen, result))

    def _make_est_output(self, gen: GenerationOutput, result) -> CoreEstimationOutput:
        p = self.params
        structures = [
            StructureResult(
                structure=s.structure,
                scores=s.scores,
                core=s.core,
                expected_deviance=s.expected_deviance,
                var_deviance=s.var_deviance,
            )
            for s in result.structures
        ]
        return CoreEstimationOutput(
            param_id=p.param_id,
            experiment_id=p.experiment_id,
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            judge_model=p.estimation.model if not self.is_synthetic else "synthetic",
            prompt_variant=gen.prompt_variant,
            prompt_text=gen.prompt_text,
            num_trajectories=gen.num_trajectories,
            total_mass=gen.total_mass,
            systems=[
                SystemResult(
                    system=p.estimation.systems[0],
                    structures=structures,
                    aggregate_core=result.aggregate_core,
                    aggregate_deviance=result.aggregate_deviance,
                )
            ],
        )

    # -------------------------------------------------------------------------
    # Save / Visualize / Summary
    # -------------------------------------------------------------------------

    def save(self):
        """Save outputs to JSON files."""
        for out in self.gen_outputs:
            path = self.output_dir / f"gen_{out.prompt_variant}.json"
            with open(path, "w") as f:
                json.dump(asdict(out), f, indent=2)

        for out in self.est_outputs:
            path = self.output_dir / f"est_{out.prompt_variant}.json"
            with open(path, "w") as f:
                json.dump(asdict(out), f, indent=2)

        print(f"\nSaved to {self.output_dir}")

    def visualize(self):
        """Generate visualizations."""
        from plot import visualize_experiment

        print("\nGenerating visualizations...")
        visualize_experiment(self.output_dir)

    def print_summary(self):
        """Print results summary."""
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        for gen, est in zip(self.gen_outputs, self.est_outputs):
            print(f"\n{gen.prompt_variant}: {gen.num_trajectories} trajectories")
            for sys in est.systems:
                for s in sys.structures:
                    print(f"  {s.structure[:35]:35} core={s.core:.3f}")
