"""Experiment runner for trajectory collection and core estimation."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

from log_utils import kv, progress, section, subsection, timer, trajectory_line
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


def _model_short(model_name: str) -> str:
    """Get short filesystem-safe model name: 'Qwen/Qwen3-1.7B' -> 'Qwen3-1.7B'."""
    return model_name.rsplit("/", 1)[-1]


class Experiment:
    """Manages experiment lifecycle: run, save, visualize."""

    def __init__(self, params: Params, output_dir: Path):
        self.params = params
        self.output_dir = output_dir
        self.gen_outputs: list[GenerationOutput] = []
        self.est_outputs: list[CoreEstimationOutput] = []
        self.est_full_outputs: list[dict] = []

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

        return cls(params, BASE_DIR / "out" / params.experiment_id)

    def run(
        self,
        max_trajectories: int | None = None,
        max_new_tokens: int | None = None,
    ) -> None:
        """Run the experiment.

        Args:
            max_trajectories: Override max trajectories per branch point.
            max_new_tokens: Override max new tokens per trajectory.
            If None, uses values from trial config.
        """
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if max_trajectories is not None:
            self.params.generation.max_trajectories = max_trajectories
        if max_new_tokens is not None:
            self.params.generation.max_new_tokens = max_new_tokens

        self._print_header()
        self._run_real()

    def _print_header(self):
        p = self.params
        section(f"EXPERIMENT: {p.experiment_id}")
        kv("Output", self.output_dir)
        kv("Generation models", len(p.generation.models))
        for i, m in enumerate(p.generation.models):
            print(f"      [{i}] {m}")
        kv("Judge model", p.estimation.model)
        kv("Prompt", repr(p.generation.prompt))
        if p.generation.base_continuation:
            kv("Base continuation", repr(p.generation.base_continuation))
        kv("Branching points", p.generation.branching_points or "(none)")
        kv("Continuation variants", len(self._build_branches()))
        kv("Structures", len(p.estimation.structures))
        for i, s in enumerate(p.estimation.structures):
            print(f"      [{i}] {s}")
        kv("Temperature", p.generation.temperature)
        kv("Top-k", p.generation.top_k)
        kv("Top-p", p.generation.top_p)
        kv("Max trajectories", p.generation.max_trajectories)
        kv("Max new tokens", p.generation.max_new_tokens)
        kv("Seed", p.generation.seed)

    def _build_branches(self) -> dict[str, str]:
        """Build dict of variant_name -> continuation text.

        The continuation is appended after the chat-templated prompt.
        Trunk gets base_continuation, branches get base_continuation + branching_point.
        """
        cont = self.params.generation.base_continuation
        branches = {"branch": cont}
        for i, branch in enumerate(self.params.generation.branching_points):
            branches[f"branch_{i}"] = cont + branch
        return branches

    # -------------------------------------------------------------------------
    # Real experiment
    # -------------------------------------------------------------------------

    def _run_real(self):
        from exploration import (
            ModelRunner,
            TrajectoryCollector,
            TrajectoryCollectorConfig,
        )
        from xenotechnics.structures.judge import JudgeStructure

        p = self.params
        branches = self._build_branches()

        n_models = len(p.generation.models)
        n_branches = len(branches)
        total_collections = n_models * n_branches
        collection_num = 0

        for model_idx, model_name in enumerate(p.generation.models):
            section(f"MODEL [{model_idx + 1}/{n_models}]: {model_name}")

            with timer(f"Loading model ({model_name})"):
                runner = ModelRunner(model_name)

            # Build formatted prompt (chat template applied to prompt)
            formatted_base = runner._apply_chat_template(p.generation.prompt)

            # Collect trajectories
            for name, continuation in branches.items():
                collection_num += 1
                variant = f"{_model_short(model_name)}_{name}"
                subsection(
                    f"Collecting [{collection_num}/{total_collections}]: {variant}"
                )
                max_traj = p.generation.max_trajectories
                kv("Model", model_name)
                kv("Prompt", repr(p.generation.prompt[:60]))
                kv(
                    "Continuation",
                    repr(continuation[:60]) if continuation else "(none)",
                )
                kv("Max trajectories", max_traj)

                config = TrajectoryCollectorConfig(
                    max_new_tokens=p.generation.max_new_tokens,
                    temperature=p.generation.temperature,
                    top_k=p.generation.top_k,
                    top_p=p.generation.top_p,
                    max_trajectories=max_traj,
                    seed=p.generation.seed,
                )

                # Track trajectory discovery via callback
                seen_count = [0]
                last_print_iter = [-1]

                def on_progress(prog):
                    new_traj = prog.trajectories_found > seen_count[0]
                    # Print on new trajectory or every 10 iterations
                    if new_traj or prog.iteration - last_print_iter[0] >= 10:
                        seen_count[0] = prog.trajectories_found
                        last_print_iter[0] = prog.iteration
                        marker = "+" if new_traj else " "
                        print(
                            f"  {marker} iter {prog.iteration:>4}  "
                            f"trajs={prog.trajectories_found:>3}  "
                            f"mass={prog.total_mass:.4f}/{prog.target_mass:.2f}  "
                            f"({prog.elapsed_seconds:.1f}s)"
                        )

                print()
                with timer("Trajectory collection"):
                    result = TrajectoryCollector(runner, config).collect(
                        formatted_base,
                        continuation=continuation,
                        progress_callback=on_progress,
                        apply_chat_template=False,
                    )

                # Print all trajectories
                print()
                for i, t in enumerate(result.trajectories):
                    trajectory_line(i + 1, t.text, t.probability, mass=None)

                # Print collection stats
                if result.stats:
                    s = result.stats
                    subsection("Collection stats")
                    kv("Stop reason", s.stop_reason)
                    kv("Unique trajectories", s.unique_trajectories)
                    kv("Duplicates", s.duplicate_trajectories)
                    kv("Failed generations", s.failed_generations)
                    kv("Total time", f"{s.total_time_seconds:.1f}s")
                    kv("Trajectories/sec", f"{s.trajectories_per_second:.1f}")
                    kv("Avg length", f"{s.avg_trajectory_length:.1f} tokens")
                    kv(
                        "Prob range",
                        f"{s.min_probability:.2e} - {s.max_probability:.2e}",
                    )

                print(
                    f"\n  Total: {len(result.trajectories)} trajectories, "
                    f"mass={result.total_mass:.4f}"
                )

                self._store_generation(
                    variant,
                    p.generation.prompt,
                    formatted_base,
                    continuation,
                    model_name,
                    result.trajectories,
                    result.total_mass,
                )

            # Free generation model before loading judge
            print(f"\n  Unloading generation model: {model_name}")
            del runner

        section("CORE ESTIMATION")
        print(f"  Total generation outputs: {len(self.gen_outputs)}")

        # Load judge model
        with timer(f"Loading judge model ({p.estimation.model})"):
            judge_runner = ModelRunner(p.estimation.model)

        # Estimate cores, capturing full judge responses
        judge_responses: dict[
            str, list[dict]
        ] = {}  # structure -> [{text, score, response}]

        def make_scorer(struct: str) -> Callable[[str], float]:
            judge = JudgeStructure(question=struct, model_runner=judge_runner)
            responses = []
            judge_responses[struct] = responses

            def scorer(text: str) -> float:
                score, raw = judge.judge(text)
                responses.append({"text": text, "score": score, "raw": raw})
                return score

            return scorer

        self._estimate_cores(make_scorer, judge_responses=judge_responses)

    def _store_generation(
        self,
        name: str,
        prompt: str,
        formatted_prompt: str,
        continuation: str,
        model: str,
        trajectories,
        total_mass: float,
    ):
        """Store generation output from collected trajectories.

        Trajectory text already includes continuation + generated tokens.
        Per-token logprobs only cover generated tokens (not continuation).
        """
        p = self.params
        records = []
        for t in trajectories:
            # per_token_logprobs only covers generated tokens (not continuation),
            # so zip from the end of the tokens tuple
            n_logprobs = len(t.per_token_logprobs)
            gen_tokens = t.tokens[-n_logprobs:] if n_logprobs else ()
            per_token_logprobs = [
                {"token": tok, "logprob": lp}
                for tok, lp in zip(gen_tokens, t.per_token_logprobs)
            ]
            records.append(
                TrajectoryRecord(
                    text=t.text,
                    probability=t.probability,
                    log_probability=t.log_probability,
                    per_token_logprobs=per_token_logprobs,
                    is_greedy=getattr(t, "is_greedy", False),
                )
            )

        gen_out = GenerationOutput(
            param_id=p.param_id,
            experiment_id=p.experiment_id,
            prompt_variant=name,
            prompt_text=prompt,
            formatted_prompt=formatted_prompt,
            continuation=continuation,
            model=model,
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            total_mass=total_mass,
            num_trajectories=len(records),
            trajectories=records,
        )
        self.gen_outputs.append(gen_out)
        print(
            f"  Stored generation: {name} ({len(records)} trajectories, mass={total_mass:.4f})"
        )

    # -------------------------------------------------------------------------
    # Core estimation
    # -------------------------------------------------------------------------

    def _estimate_cores(
        self,
        make_scorer: Callable[[str], Callable[[str], float]],
        judge_responses: dict[str, list[dict]] | None = None,
    ):
        from exploration import CoreEstimator, CoreEstimatorConfig

        p = self.params
        estimator = CoreEstimator(CoreEstimatorConfig())
        n_structures = len(p.estimation.structures)

        n_gen = len(self.gen_outputs)
        for gen_idx, gen in enumerate(self.gen_outputs):
            subsection(
                f"Estimating cores [{gen_idx + 1}/{n_gen}]: {gen.prompt_variant}"
            )
            kv("Model", gen.model)
            kv("Prompt", repr(gen.prompt_text[:80]))
            kv("Trajectories", gen.num_trajectories)
            kv("Total mass", f"{gen.total_mass:.4f}")
            kv("Structures", n_structures)

            # judge_responses gets fresh lists per structure from make_scorer
            print()
            with timer("Core estimation"):
                result = estimator.estimate(
                    gen.trajectories,
                    p.estimation.structures,
                    make_scorer,
                )

            # Print per-structure results
            print()
            for i, s in enumerate(result.structures):
                progress(i + 1, n_structures, s.structure[:50])
                n_positive = sum(1 for v in s.scores if v > 0.5)
                kv(
                    "Scores",
                    f"{n_positive}/{len(s.scores)} positive",
                    indent=6,
                )
                kv("Core", f"{s.core:.4f}", indent=6)
                kv("Deviance", f"{s.expected_deviance:.4f}", indent=6)

            print(f"\n  Aggregate core: {result.aggregate_core:.4f}")
            print(f"  Aggregate deviance: {result.aggregate_deviance:.4f}")

            self.est_outputs.append(self._make_est_output(gen, result))

            # Store full judge responses for this generation
            if judge_responses is not None:
                self.est_full_outputs.append(
                    self._make_est_full_output(gen, judge_responses)
                )

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
            judge_model=p.estimation.model,
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

    def _make_est_full_output(
        self, gen: GenerationOutput, judge_responses: dict[str, list[dict]]
    ) -> dict:
        """Build est_full dict with raw judge responses per trajectory."""
        p = self.params
        structures = []
        for struct_name, responses in judge_responses.items():
            structures.append(
                {
                    "structure": struct_name,
                    "judgments": [
                        {
                            "trajectory_text": r["text"],
                            "score": r["score"],
                            "judge_output": r["raw"],
                        }
                        for r in responses
                    ],
                }
            )
        return {
            "param_id": p.param_id,
            "experiment_id": p.experiment_id,
            "prompt_variant": gen.prompt_variant,
            "judge_model": p.estimation.model,
            "num_trajectories": gen.num_trajectories,
            "structures": structures,
        }

    # -------------------------------------------------------------------------
    # Save / Visualize / Summary
    # -------------------------------------------------------------------------

    def _variant_dir(self, variant: str, model: str) -> Path:
        """Get output subdirectory for a model+variant combo."""
        model_short = _model_short(model)
        # Strip model prefix from variant to get branch name
        branch = variant
        if branch.startswith(model_short + "_"):
            branch = branch[len(model_short) + 1 :]
        # Rename bare "branch" to "trunk"
        if branch == "branch":
            branch = "trunk"
        return self.output_dir / model_short / branch

    def save(self):
        """Save outputs to JSON files."""
        subsection("Saving results")
        for out in self.gen_outputs:
            d = self._variant_dir(out.prompt_variant, out.model)
            d.mkdir(parents=True, exist_ok=True)
            path = d / "gen.json"
            with open(path, "w") as f:
                json.dump(asdict(out), f, indent=2)
            print(
                f"  Wrote {path.relative_to(self.output_dir)} ({out.num_trajectories} trajectories)"
            )

        for out, gen in zip(self.est_outputs, self.gen_outputs):
            d = self._variant_dir(out.prompt_variant, gen.model)
            d.mkdir(parents=True, exist_ok=True)
            path = d / "est.json"
            with open(path, "w") as f:
                json.dump(asdict(out), f, indent=2)
            print(f"  Wrote {path.relative_to(self.output_dir)}")

        for out, gen in zip(self.est_full_outputs, self.gen_outputs):
            d = self._variant_dir(out["prompt_variant"], gen.model)
            d.mkdir(parents=True, exist_ok=True)
            path = d / "est_full.json"
            with open(path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"  Wrote {path.relative_to(self.output_dir)}")

        print(f"\n  All saved to {self.output_dir}")

    def visualize(self, max_viz_samples_per_branch_point: int | None = None):
        """Generate visualizations."""
        from plot import visualize_experiment

        subsection("Generating visualizations")
        visualize_experiment(
            self.output_dir,
            max_viz_samples_per_branch_point=max_viz_samples_per_branch_point,
        )

    def print_summary(self):
        """Print results summary."""
        section("SUMMARY")

        for gen, est in zip(self.gen_outputs, self.est_outputs):
            subsection(
                f"{gen.prompt_variant}: {gen.num_trajectories} trajectories, "
                f"mass={gen.total_mass:.4f}"
            )
            kv("Model", gen.model)
            kv("Prompt", repr(gen.prompt_text[:60]))
            for sys in est.systems:
                for s in sys.structures:
                    n_positive = sum(1 for v in s.scores if v > 0.5)
                    kv(
                        s.structure[:45],
                        f"core={s.core:.3f}  dev={s.expected_deviance:.3f}  "
                        f"({n_positive}/{len(s.scores)} positive)",
                    )
