#!/usr/bin/env python3
"""
Continuation Scorer - Compute P(continuation | prompt) for language models.

Uses Scorer from tools/ for local models and
CloudScorerGenerator for cloud inference.

Usage: python tools/continuation_scorer.py [config.json] [--profile] [--isolate]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG = "tools/sample_configs/continuation_scorer.json"


# -----------------------------------------------------------------------------
# Input/Output Data Structures
# -----------------------------------------------------------------------------


@dataclass
class ContinuationScorerInput:
    """Input for continuation scoring: prompt, continuations, models."""

    prompt: str | None
    continuations: list[str]
    models: list[str]
    cloud_models: list[str] | None = None  # Models to run via cloud API

    @classmethod
    def from_json(cls, path: Path) -> ContinuationScorerInput:
        with open(path) as f:
            d = json.load(f)
        return cls(
            d.get("prompt"),
            d.get("continuations", []),
            d["models"],
            d.get("cloud"),  # Optional cloud models list
        )


@dataclass
class Score:
    """Token/continuation probability score."""

    text: str
    prob: float
    logprob: float
    num_tokens: int = 1


@dataclass
class ModelResult:
    """Per-model results: greedy predictions and continuation scores."""

    greedy_next: Score
    greedy_trajectory: Score
    scores: dict[str, Score] = field(default_factory=dict)


@dataclass
class ModelProfile:
    """Timing profile for a single model."""

    model: str
    load_time: float
    inference_time: float
    total_time: float

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "load_time": round(self.load_time, 2),
            "inference_time": round(self.inference_time, 2),
            "total_time": round(self.total_time, 2),
        }


@dataclass
class ContinuationScorerOutput:
    """Output from continuation scoring."""

    prompt: str | None
    results: dict[str, ModelResult]
    continuations: list[str]
    profiles: list[ModelProfile] | None = None

    def to_dict(self) -> dict:
        """Format as JSON-serializable dict."""
        out: dict = {"prompt": self.prompt or "<BOS>", "models": {}}

        for name, r in self.results.items():
            out["models"][name] = {
                "greedy_next": {"text": r.greedy_next.text, "prob": r.greedy_next.prob},
                "greedy_trajectory": {
                    "text": r.greedy_trajectory.text,
                    "prob": r.greedy_trajectory.prob,
                    "tokens": r.greedy_trajectory.num_tokens,
                },
            }

        if self.continuations:
            out["continuations"] = []
            for c in self.continuations:
                scores = {n: r.scores[c] for n, r in self.results.items()}
                out["continuations"].append(
                    {
                        "text": c,
                        "by_model": {
                            n: {"prob": s.prob, "logprob": s.logprob}
                            for n, s in scores.items()
                        },
                        "avg_prob": sum(s.prob for s in scores.values()) / len(scores),
                        "avg_logprob": sum(s.logprob for s in scores.values())
                        / len(scores),
                    }
                )
            if len(self.continuations) > 1:
                out["ranking"] = sorted(
                    [(c["text"], c["avg_prob"]) for c in out["continuations"]],
                    key=lambda x: -x[1],
                )

        if self.profiles:
            out["profile"] = {
                "models": [p.to_dict() for p in self.profiles],
                "total_time": round(sum(p.total_time for p in self.profiles), 2),
            }

        return out


# -----------------------------------------------------------------------------
# Scoring with Generators
# -----------------------------------------------------------------------------


def score_model_local(
    model_name: str,
    prompt: str | None,
    continuations: list[str],
    max_tokens: int = 20,
    isolate: bool = False,
    verbose: bool = True,
    debug: bool = False,
    use_chat_template: bool = True,
) -> tuple[ModelResult, float, float]:
    """
    Score continuations using Scorer.

    Returns:
        (ModelResult, load_time, inference_time)
    """
    from exploration.common import ModelWrapper
    from exploration.generators import GreedyGenerator
    from tools.common import Scorer
    from xenotechnics.common import String

    t_start = time.time()

    # Load model once and share between generators
    if not isolate:
        model = ModelWrapper(model_name=model_name)
    else:
        model = None

    # Create greedy generator and scorer sharing the same model
    greedy_gen = GreedyGenerator(
        model_name=model_name,
        use_chat_template=use_chat_template,
        lazy_load=isolate,
        model=model,
        debug=debug,
    )
    scorer = Scorer(
        model_name=model_name,
        use_chat_template=use_chat_template,
        model=model,
        debug=debug,
    )

    t_load = time.time()
    load_time = t_load - t_start

    if verbose:
        print(f"  Model loaded in {load_time:.2f}s", flush=True)

    # Prepare prompt
    prompt_string = String.from_text(prompt) if prompt else None

    # Greedy next token
    if verbose:
        print("  Greedy next...", flush=True)

    tree = greedy_gen.run(
        prompt=prompt_string,
        max_new_tokens=1,
        verbose=False,
        isolate=isolate,
    )

    # Get the generated token and its probability from the tree
    trajectories = tree.get_trajectory_nodes()
    if trajectories:
        traj = trajectories[0]
        # Get continuation (after prompt)
        path = []
        node = traj
        while node.parent is not None:
            path.append(node)
            node = node.parent
        path.reverse()

        # Find first non-prompt token and its probability
        greedy_token = ""
        greedy_logprob = 0.0
        if path:
            last = path[-1]
            greedy_token = last.string.tokens[-1] if last.string.tokens else ""
            parent = last.parent
            if parent and greedy_token in parent.child_logprobs:
                greedy_logprob = parent.child_logprobs[greedy_token]

        greedy_next = Score(
            text=greedy_token,
            prob=math.exp(greedy_logprob) if greedy_logprob > float("-inf") else 0.0,
            logprob=greedy_logprob,
        )
    else:
        greedy_next = Score(text="", prob=0.0, logprob=float("-inf"))

    if verbose:
        print(f"    '{greedy_next.text}' p={greedy_next.prob:.4f}", flush=True)

    # Greedy trajectory
    if verbose:
        print("  Greedy trajectory...", flush=True)

    tree = greedy_gen.run(
        prompt=prompt_string,
        max_new_tokens=max_tokens,
        verbose=False,
        isolate=isolate,
    )

    trajectories = tree.get_trajectory_nodes()
    if trajectories:
        traj = trajectories[0]
        text = traj.string.to_text()
        # Remove prompt from text
        if prompt:
            text = text[len(prompt) :] if text.startswith(prompt) else text

        # Get continuation logprob
        prompt_len = len(prompt_string.tokens) if prompt_string else 0
        cont_logprob = traj.get_continuation_logprob(prompt_len)

        greedy_traj = Score(
            text=text,
            prob=math.exp(cont_logprob) if cont_logprob > float("-inf") else 0.0,
            logprob=cont_logprob,
            num_tokens=len(traj.string.tokens) - prompt_len if traj.string else 0,
        )
    else:
        greedy_traj = Score(text="", prob=0.0, logprob=float("-inf"), num_tokens=0)

    if verbose:
        text_preview = (
            greedy_traj.text[:40] + "..."
            if len(greedy_traj.text) > 40
            else greedy_traj.text
        )
        print(f"    '{text_preview}' ({greedy_traj.num_tokens} tok)", flush=True)

    # Score continuations
    scores = {}
    if continuations:
        if verbose:
            print("  Continuations...", flush=True)

        for cont in continuations:
            result = scorer.score(
                prompt=prompt_string,
                continuation=cont,
                verbose=False,
            )

            scores[cont] = Score(
                text=cont,
                prob=result["prob"],
                logprob=result["logprob"],
                num_tokens=result["n_tokens"],
            )

            if verbose:
                print(
                    f"    P({cont!r}) = {scores[cont].prob:.4e} ({scores[cont].num_tokens} tok)",
                    flush=True,
                )

    t_end = time.time()
    inference_time = t_end - t_load

    return ModelResult(greedy_next, greedy_traj, scores), load_time, inference_time


def score_model_cloud(
    model_name: str,
    prompt: str | None,
    continuations: list[str],
    max_tokens: int = 20,
    verbose: bool = True,
) -> tuple[ModelResult, float]:
    """
    Score continuations using CloudScorerGenerator.

    Returns:
        (ModelResult, total_time)
    """
    from exploration.generators import CloudGreedyGenerator, CloudScorerGenerator

    t_start = time.time()

    greedy_gen = CloudGreedyGenerator(model_name)
    scorer_gen = CloudScorerGenerator(model_name)

    # Greedy next token
    if verbose:
        print("  [CLOUD] Greedy next...", flush=True)

    result = greedy_gen.greedy_next(prompt or "", verbose=False)
    greedy_next = Score(
        text=result.text,
        prob=result.prob,
        logprob=result.logprob,
    )

    if verbose:
        print(f"    '{greedy_next.text}' p={greedy_next.prob:.4f}", flush=True)

    # Greedy trajectory
    if verbose:
        print("  [CLOUD] Greedy trajectory...", flush=True)

    result = greedy_gen.generate(prompt or "", max_new_tokens=max_tokens, verbose=False)
    greedy_traj = Score(
        text=result.text,
        prob=result.prob,
        logprob=result.logprob,
        num_tokens=result.n_tokens,
    )

    if verbose:
        text_preview = (
            greedy_traj.text[:40] + "..."
            if len(greedy_traj.text) > 40
            else greedy_traj.text
        )
        print(f"    '{text_preview}' ({greedy_traj.num_tokens} tok)", flush=True)

    # Score continuations
    scores = {}
    if continuations:
        if verbose:
            print("  [CLOUD] Continuations...", flush=True)

        for cont in continuations:
            result = scorer_gen.score(prompt or "", cont, verbose=False)

            scores[cont] = Score(
                text=cont,
                prob=result["prob"],
                logprob=result["logprob"],
                num_tokens=result["n_tokens"],
            )

            if verbose:
                print(
                    f"    P({cont!r}) = {scores[cont].prob:.4e} ({scores[cont].num_tokens} tok)",
                    flush=True,
                )

    total_time = time.time() - t_start
    return ModelResult(greedy_next, greedy_traj, scores), total_time


# -----------------------------------------------------------------------------
# Core Logic
# -----------------------------------------------------------------------------


def score_continuations(
    inp: ContinuationScorerInput,
    max_tokens: int = 20,
    profile: bool = False,
    isolate: bool = False,
    debug: bool = False,
    use_chat_template: bool = True,
) -> ContinuationScorerOutput:
    """Main scoring logic: score all continuations with all models."""
    results = {}
    profiles = [] if profile else None

    # Local models
    for i, name in enumerate(inp.models, 1):
        if profile:
            print(f"\n[PROFILE {i}/{len(inp.models)}] {name}", flush=True)
        else:
            print(f"\n{name}", flush=True)

        result, load_time, inference_time = score_model_local(
            model_name=name,
            prompt=inp.prompt,
            continuations=inp.continuations,
            max_tokens=max_tokens,
            isolate=isolate,
            verbose=True,
            debug=debug,
            use_chat_template=use_chat_template,
        )

        results[name] = result

        if profile:
            total_time = load_time + inference_time
            print(f"  LOAD: {load_time:.2f}s", flush=True)
            print(f"  INFERENCE: {inference_time:.2f}s", flush=True)
            print(f"  TOTAL: {total_time:.2f}s", flush=True)
            profiles.append(ModelProfile(name, load_time, inference_time, total_time))

    # Cloud models
    if inp.cloud_models:
        for name in inp.cloud_models:
            if profile:
                print(f"\n[CLOUD] {name}", flush=True)
            else:
                print(f"\n[CLOUD] {name}", flush=True)

            result, total_time = score_model_cloud(
                model_name=name,
                prompt=inp.prompt,
                continuations=inp.continuations,
                max_tokens=max_tokens,
                verbose=True,
            )

            results[name] = result

            if profile:
                print(f"  TOTAL: {total_time:.2f}s", flush=True)
                profiles.append(ModelProfile(name, 0.0, total_time, total_time))

    return ContinuationScorerOutput(
        prompt=inp.prompt,
        results=results,
        continuations=inp.continuations,
        profiles=profiles,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG, help="Config JSON")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling: show load/inference time per model",
    )
    parser.add_argument(
        "--isolate",
        action="store_true",
        help="Run each model in isolated subprocess (clean memory between runs)",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug output (token ID, probability, text for each step)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate for greedy trajectory (default: 100)",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable chat template (use raw text completion instead)",
    )
    return parser.parse_args()


def input_from_args(args: argparse.Namespace) -> ContinuationScorerInput:
    """Load input from command line arguments."""
    path = Path(args.config)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return ContinuationScorerInput.from_json(path)


def print_output(output: ContinuationScorerOutput) -> None:
    """Print output to stdout."""
    print("\n" + "=" * 50 + "\nRESULTS\n" + "=" * 50 + "\n")
    print(json.dumps(output.to_dict(), indent=2))


def print_profile_summary(output: ContinuationScorerOutput) -> None:
    """Print a profile summary table."""
    if not output.profiles:
        return

    print("\n" + "=" * 70)
    print("PROFILE SUMMARY")
    print("=" * 70)
    print(f"{'Model':<45} {'Load':>8} {'Infer':>8} {'Total':>8}")
    print("-" * 70)

    for p in output.profiles:
        name = p.model if len(p.model) <= 43 else "..." + p.model[-40:]
        print(
            f"{name:<45} {p.load_time:>7.2f}s {p.inference_time:>7.2f}s {p.total_time:>7.2f}s"
        )

    total = sum(p.total_time for p in output.profiles)
    print("-" * 70)
    print(f"{'TOTAL':<45} {'':<8} {'':<8} {total:>7.2f}s")
    print("=" * 70)


def main() -> int:
    args = get_args()

    # Print config info
    path = Path(args.config)
    if not path.exists():
        return print(f"Error: {path} not found") or 1

    inp: ContinuationScorerInput = input_from_args(args)

    print(f"Prompt: {f'{inp.prompt!r}' if inp.prompt else '<BOS>'}")
    if inp.continuations:
        print(f"Continuations: {inp.continuations}")
    print(f"Models (local): {inp.models}")
    if inp.cloud_models:
        print(f"Models (cloud): {inp.cloud_models}")
    if args.profile:
        print("Profiling: ENABLED")
    if args.isolate:
        print("Isolation: ENABLED (subprocess per model)")
    debug = not args.no_debug
    if not debug:
        print("Debug: DISABLED")
    use_chat_template = not args.no_chat_template
    print(f"Chat template: {'ENABLED' if use_chat_template else 'DISABLED'}")
    print(f"Max tokens: {args.max_tokens}")
    print("-" * 50)

    output = score_continuations(
        inp,
        max_tokens=args.max_tokens,
        profile=args.profile,
        isolate=args.isolate,
        debug=debug,
        use_chat_template=use_chat_template,
    )

    if args.profile:
        print_profile_summary(output)

    print_output(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
