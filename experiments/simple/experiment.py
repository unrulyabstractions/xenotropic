"""Experiment runner for trajectory collection and core estimation."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

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

SCRIPT_DIR = Path(__file__).parent


class Experiment:
    """Manages experiment lifecycle: run, save, visualize."""

    def __init__(self, params: Params, output_dir: Path, synthetic: bool = False):
        self.params = params
        self.output_dir = output_dir
        self.synthetic = synthetic
        self.gen_outputs: list[GenerationOutput] = []
        self.est_outputs: list[CoreEstimationOutput] = []

    @classmethod
    def from_trial(cls, trial_name: str, synthetic: bool = False) -> Experiment:
        """Load experiment from trial config file."""
        trial_path = SCRIPT_DIR / "trials" / f"{trial_name}.json"
        if not trial_path.exists():
            available = [p.stem for p in (SCRIPT_DIR / "trials").glob("*.json")]
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

        suffix = "_synthetic" if synthetic else ""
        output_dir = SCRIPT_DIR / "out" / f"{trial_name}{suffix}"

        return cls(params, output_dir, synthetic)

    def run(self, **kwargs) -> None:
        """Run the experiment."""
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Output: {self.output_dir}")
        self._print_header()

        if self.synthetic:
            self._run_synthetic(**kwargs)
        else:
            self._run_real(**kwargs)

    def _print_header(self):
        p = self.params
        mode = "SYNTHETIC" if self.synthetic else ""
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
        self, target_mass: float = 0.9, max_iterations: int = 200, verbose: bool = True
    ):
        from exploration import (
            CoreEstimator,
            CoreEstimatorConfig,
            ModelRunner,
            TrajectoryCollector,
            TrajectoryCollectorConfig,
        )
        from xenotechnics.structures.judge import JudgeStructure

        p = self.params
        prompts = self._build_prompts()

        print("Loading model...")
        runner = ModelRunner(p.generation.model)

        for name, text in prompts.items():
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
            result = TrajectoryCollector(runner, config).collect(text)

            trajectories = [
                TrajectoryRecord(
                    text=t.text,
                    probability=t.probability,
                    log_probability=t.log_probability,
                    per_token_logprobs=[
                        {"token": tok, "logprob": lp}
                        for tok, lp in zip(t.tokens, t.per_token_logprobs)
                    ],
                )
                for t in result.trajectories
            ]

            self.gen_outputs.append(
                GenerationOutput(
                    param_id=p.param_id,
                    experiment_id=p.experiment_id,
                    prompt_variant=name,
                    prompt_text=text,
                    model=p.generation.model,
                    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                    total_mass=result.total_mass,
                    num_trajectories=len(trajectories),
                    trajectories=trajectories,
                )
            )

            if verbose:
                print(
                    f"  {len(trajectories)} trajectories, mass={result.total_mass:.3f}"
                )

        # Estimate cores
        estimator = CoreEstimator(CoreEstimatorConfig(use_log_space=True))

        for gen in self.gen_outputs:
            if verbose:
                print(f"\nEstimating cores: {gen.prompt_variant}")

            def make_scorer(struct):
                judge = JudgeStructure(question=struct, model_runner=runner)
                return lambda text: judge.judge(text)[0]

            result = estimator.estimate(
                gen.trajectories, p.estimation.structures, make_scorer, gen.prompt_text
            )

            self.est_outputs.append(
                self._make_est_output(gen, result, p.estimation.model)
            )

    # -------------------------------------------------------------------------
    # Synthetic experiment
    # -------------------------------------------------------------------------

    def _run_synthetic(
        self,
        num_trajectories: Optional[int] = None,
        seed: int = 42,
        verbose: bool = True,
    ):
        from exploration import CoreEstimator, CoreEstimatorConfig

        p = self.params
        prompts = self._build_prompts()
        num_traj = num_trajectories or p.generation.max_trajectories or 20

        gen = _SyntheticGenerator(seed)
        scorer = _SyntheticScorer(seed + 1000)

        for name, text in prompts.items():
            if verbose:
                print(f"\nGenerating: {name}")

            trajectories, mass = gen.generate(text, num_traj)

            self.gen_outputs.append(
                GenerationOutput(
                    param_id=p.param_id,
                    experiment_id=p.experiment_id,
                    prompt_variant=name,
                    prompt_text=text,
                    model=p.generation.model + " (synthetic)",
                    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                    total_mass=mass,
                    num_trajectories=len(trajectories),
                    trajectories=trajectories,
                )
            )

            if verbose:
                print(f"  {len(trajectories)} trajectories, mass={mass:.3f}")

        # Estimate cores
        estimator = CoreEstimator(CoreEstimatorConfig(use_log_space=False))

        for gen_out in self.gen_outputs:
            if verbose:
                print(f"\nEstimating cores: {gen_out.prompt_variant}")

            def make_scorer(struct):
                return lambda text: scorer.score(text, struct)

            result = estimator.estimate(
                gen_out.trajectories, p.estimation.structures, make_scorer, ""
            )

            self.est_outputs.append(
                self._make_est_output(
                    gen_out, result, p.estimation.model + " (synthetic)"
                )
            )

    def _make_est_output(self, gen, result, judge_model) -> CoreEstimationOutput:
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
            judge_model=judge_model,
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
        from visualize_experiment import visualize_results

        viz_dir = self.output_dir / "viz"
        print("\nGenerating visualizations...")
        visualize_results(self.output_dir, viz_dir)

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


# -----------------------------------------------------------------------------
# Synthetic helpers
# -----------------------------------------------------------------------------


class _SyntheticGenerator:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def generate(self, prompt: str, n: int):
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
        ]
        ranks = np.arange(1, len(continuations) + 1)
        probs = 1.0 / (ranks**1.2)
        probs = probs * (1 + 0.1 * self.rng.standard_normal(len(probs)))
        probs = np.maximum(probs, 0.001)
        probs = probs / probs.sum()

        n = min(n, len(continuations))
        trajectories = []
        mass = 0.0

        for i in range(n):
            p = float(probs[i])
            mass += p
            trajectories.append(
                TrajectoryRecord(
                    text=prompt + continuations[i],
                    probability=p,
                    log_probability=float(np.log(p + 1e-10)),
                    per_token_logprobs=[],
                )
            )

        return trajectories, mass


class _SyntheticScorer:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def score(self, text: str, structure: str) -> float:
        h = hash(text + structure) % 1000
        return float(
            np.clip((h / 1000) * 0.6 + 0.2 + self.rng.standard_normal() * 0.1, 0, 1)
        )
