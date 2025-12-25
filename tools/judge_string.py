#!/usr/bin/env python3
"""
Judge String - Ask LLMs to score a string on a 0-1 scale for multiple questions.

Uses JudgeStructure from xenotechnics/structures/judge.py as the interface.

Usage: python tools/judge_string.py [config.json] [--viz] [--profile] [--isolate]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_CONFIG = "tools/sample_configs/judge_string.json"


# -----------------------------------------------------------------------------
# Input/Output Data Structures
# -----------------------------------------------------------------------------


@dataclass
class JudgeStringInput:
    """Input for string judging: string, questions, models."""

    string: str
    questions: list[str]
    models: list[str]
    visualize: bool = False
    cloud_models: list[str] | None = None

    @classmethod
    def from_json(cls, path: Path, visualize: bool = False) -> JudgeStringInput:
        with open(path) as f:
            d = json.load(f)
        questions = d.get("questions", [d["question"]] if "question" in d else [])
        return cls(
            d["string"],
            questions,
            d["models"],
            visualize,
            d.get("cloud"),
        )


@dataclass
class QuestionJudgment:
    """Judgment for a single question from a single model."""

    raw_response: str
    parsed_score: float | None
    success: bool


@dataclass
class ModelResults:
    """All judgments from a single model."""

    model_name: str
    judgments: dict[str, QuestionJudgment] = field(default_factory=dict)

    def get_scores(self) -> dict[str, float]:
        """Get successfully parsed scores by question."""
        return {
            q: j.parsed_score
            for q, j in self.judgments.items()
            if j.success and j.parsed_score is not None
        }

    def score_list(self) -> list[float]:
        """Get list of successful scores."""
        return list(self.get_scores().values())


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
class QuestionStats:
    """Statistics for a single question across all models."""

    question: str
    scores: dict[str, float]
    mean: float
    std: float
    min_val: float
    max_val: float
    range_val: float
    n_models: int


@dataclass
class ModelStats:
    """Statistics for a single model across all questions."""

    model_name: str
    scores: dict[str, float]
    mean: float
    std: float
    min_val: float
    max_val: float
    range_val: float
    n_questions: int


@dataclass
class CrossAnalysis:
    """Cross-model and cross-question analysis."""

    question_stats: dict[str, QuestionStats]
    model_stats: dict[str, ModelStats]
    model_agreement: float
    question_discrimination: float
    score_matrix: np.ndarray
    model_names: list[str]
    question_labels: list[str]


@dataclass
class JudgeStringOutput:
    """Output from string judging."""

    string: str
    questions: list[str]
    model_results: dict[str, ModelResults] = field(default_factory=dict)
    analysis: CrossAnalysis | None = None
    profiles: list[ModelProfile] | None = None

    def to_dict(self) -> dict:
        """Format as JSON-serializable dict."""
        out: dict = {
            "string": self.string,
            "questions": self.questions,
            "models": {},
        }

        for model_name, results in self.model_results.items():
            out["models"][model_name] = {
                "judgments": {
                    q: {
                        "raw_response": j.raw_response,
                        "parsed_score": j.parsed_score,
                        "success": j.success,
                    }
                    for q, j in results.judgments.items()
                },
                "scores": results.get_scores(),
            }

        if self.analysis:
            out["analysis"] = {
                "question_stats": {
                    q: {
                        "mean": s.mean,
                        "std": s.std,
                        "min": s.min_val,
                        "max": s.max_val,
                        "range": s.range_val,
                        "scores_by_model": s.scores,
                    }
                    for q, s in self.analysis.question_stats.items()
                },
                "model_stats": {
                    m: {
                        "mean": s.mean,
                        "std": s.std,
                        "min": s.min_val,
                        "max": s.max_val,
                        "range": s.range_val,
                        "scores_by_question": s.scores,
                    }
                    for m, s in self.analysis.model_stats.items()
                },
                "model_agreement": self.analysis.model_agreement,
                "question_discrimination": self.analysis.question_discrimination,
            }

        if self.profiles:
            out["profile"] = {
                "models": [p.to_dict() for p in self.profiles],
                "total_time": round(sum(p.total_time for p in self.profiles), 2),
            }

        return out


# -----------------------------------------------------------------------------
# Judging with JudgeStructure
# -----------------------------------------------------------------------------


def judge_model(
    model_name: str,
    string: str,
    questions: list[str],
    use_cloud: bool = False,
    isolate: bool = False,
    verbose: bool = True,
) -> tuple[ModelResults, float, float]:
    """
    Judge string with model using JudgeStructure.

    Returns:
        (ModelResults, load_time, inference_time)
    """
    from xenotechnics.structures.judge import JudgeStructure

    t_start = time.time()

    # Create judges for each question (they share the same generator)
    judges = {}
    for question in questions:
        judges[question] = JudgeStructure(
            question=question,
            model_name=model_name,
            use_cloud=use_cloud,
            isolate=isolate,
        )

    # Force load the first judge's generator to measure load time
    if questions and not use_cloud:
        _ = judges[questions[0]].generator

    t_load = time.time()
    load_time = t_load - t_start

    if verbose:
        print(f"  Model loaded in {load_time:.2f}s", flush=True)

    model_results = ModelResults(model_name=model_name)

    for question in questions:
        if verbose:
            print(f"\n  Q: {question[:50]}{'...' if len(question) > 50 else ''}")

        judge = judges[question]
        score, raw_response = judge.judge(string)

        # Determine success based on whether score is the default 0.5
        # A more robust check: if raw response contains a number, it succeeded
        import re

        has_number = bool(re.search(r"\d+\.?\d*", raw_response))
        success = has_number or score != 0.5

        judgment = QuestionJudgment(
            raw_response=raw_response,
            parsed_score=score if success else None,
            success=success,
        )

        model_results.judgments[question] = judgment

        if verbose:
            status = "\u2713" if judgment.success else "\u2717"
            score_str = (
                f"{judgment.parsed_score:.3f}"
                if judgment.parsed_score is not None
                else "N/A"
            )
            print(f"     {status} '{raw_response}' -> {score_str}")

    t_end = time.time()
    inference_time = t_end - t_load

    return model_results, load_time, inference_time


# -----------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------


def compute_analysis(
    model_results: dict[str, ModelResults],
    questions: list[str],
) -> CrossAnalysis:
    """Compute cross-model and cross-question statistics."""
    model_names = list(model_results.keys())
    n_models = len(model_names)
    n_questions = len(questions)

    # Build score matrix
    score_matrix = np.full((n_models, n_questions), np.nan)
    for i, model_name in enumerate(model_names):
        scores = model_results[model_name].get_scores()
        for j, q in enumerate(questions):
            if q in scores:
                score_matrix[i, j] = scores[q]

    # Per-question stats
    question_stats = {}
    for j, q in enumerate(questions):
        col = score_matrix[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 0:
            scores_by_model = {
                model_names[i]: float(col[i])
                for i in range(n_models)
                if not np.isnan(col[i])
            }
            question_stats[q] = QuestionStats(
                question=q,
                scores=scores_by_model,
                mean=float(np.mean(valid)),
                std=float(np.std(valid)) if len(valid) > 1 else 0.0,
                min_val=float(np.min(valid)),
                max_val=float(np.max(valid)),
                range_val=float(np.max(valid) - np.min(valid)),
                n_models=len(valid),
            )

    # Per-model stats
    model_stats = {}
    for i, model_name in enumerate(model_names):
        row = score_matrix[i, :]
        valid = row[~np.isnan(row)]
        if len(valid) > 0:
            scores_by_question = {
                questions[j]: float(row[j])
                for j in range(n_questions)
                if not np.isnan(row[j])
            }
            model_stats[model_name] = ModelStats(
                model_name=model_name,
                scores=scores_by_question,
                mean=float(np.mean(valid)),
                std=float(np.std(valid)) if len(valid) > 1 else 0.0,
                min_val=float(np.min(valid)),
                max_val=float(np.max(valid)),
                range_val=float(np.max(valid) - np.min(valid)),
                n_questions=len(valid),
            )

    # Model agreement
    model_agreement = 0.0
    if n_models > 1:
        correlations = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                row_i = score_matrix[i, :]
                row_j = score_matrix[j, :]
                valid_mask = ~np.isnan(row_i) & ~np.isnan(row_j)
                if np.sum(valid_mask) > 1:
                    corr = np.corrcoef(row_i[valid_mask], row_j[valid_mask])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
        if correlations:
            model_agreement = float(np.mean(correlations))

    # Question discrimination
    question_means = [s.mean for s in question_stats.values()]
    question_discrimination = (
        float(np.var(question_means)) if len(question_means) > 1 else 0.0
    )

    question_labels = [q[:30] + "..." if len(q) > 30 else q for q in questions]

    return CrossAnalysis(
        question_stats=question_stats,
        model_stats=model_stats,
        model_agreement=model_agreement,
        question_discrimination=question_discrimination,
        score_matrix=score_matrix,
        model_names=model_names,
        question_labels=question_labels,
    )


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------


def create_visualizations(output: JudgeStringOutput, save_path: Path | None = None):
    """Create visualizations of the results."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if output.analysis is None:
        print("No analysis data for visualization.")
        return

    analysis = output.analysis
    n_models = len(analysis.model_names)
    n_questions = len(analysis.question_labels)

    COLORS = {
        "primary": "#2E4057",
        "neutral": "#8D99AE",
        "background": "#FAFBFC",
        "text": "#1A1A2E",
    }

    score_cmap = LinearSegmentedColormap.from_list(
        "score", ["#D32F2F", "#FFEB3B", "#2E7D32"]
    )

    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["background"],
            "axes.facecolor": "white",
            "axes.labelcolor": COLORS["text"],
            "axes.titleweight": "bold",
            "font.family": "sans-serif",
            "font.size": 10,
        }
    )

    if n_questions == 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax_heatmap = axes[0]
        ax_model_bars = axes[1]
        ax_question_bars = None
        ax_summary = None
    elif n_models == 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax_heatmap = axes[0]
        ax_question_bars = axes[1]
        ax_model_bars = None
        ax_summary = None
    else:
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax_heatmap = fig.add_subplot(gs[0, 0])
        ax_model_bars = fig.add_subplot(gs[0, 1])
        ax_question_bars = fig.add_subplot(gs[1, 0])
        ax_summary = fig.add_subplot(gs[1, 1])

    # Heatmap
    matrix = analysis.score_matrix
    im = ax_heatmap.imshow(matrix, cmap=score_cmap, aspect="auto", vmin=0, vmax=1)

    ax_heatmap.set_xticks(range(n_questions))
    ax_heatmap.set_xticklabels(
        analysis.question_labels, rotation=45, ha="right", fontsize=8
    )
    ax_heatmap.set_yticks(range(n_models))
    ax_heatmap.set_yticklabels(
        [m.split("/")[-1] for m in analysis.model_names], fontsize=9
    )

    for i in range(n_models):
        for j in range(n_questions):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if 0.3 < val < 0.7 else "black"
                ax_heatmap.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=color,
                )

    ax_heatmap.set_title("Score Matrix", fontsize=12, pad=10)
    plt.colorbar(im, ax=ax_heatmap, shrink=0.8)

    # Model bars
    if ax_model_bars is not None and analysis.model_stats:
        model_means = [
            analysis.model_stats[m].mean
            for m in analysis.model_names
            if m in analysis.model_stats
        ]
        model_stds = [
            analysis.model_stats[m].std
            for m in analysis.model_names
            if m in analysis.model_stats
        ]
        model_labels = [
            m.split("/")[-1] for m in analysis.model_names if m in analysis.model_stats
        ]

        y_pos = np.arange(len(model_labels))
        colors = [score_cmap(m) for m in model_means]

        ax_model_bars.barh(
            y_pos, model_means, xerr=model_stds, capsize=4, color=colors, alpha=0.9
        )
        ax_model_bars.set_yticks(y_pos)
        ax_model_bars.set_yticklabels(model_labels, fontsize=9)
        ax_model_bars.set_xlim(0, 1)
        ax_model_bars.set_title("Model Comparison", fontsize=12, pad=10)

    # Question bars
    if ax_question_bars is not None and analysis.question_stats:
        question_means = [
            analysis.question_stats[q].mean
            for q in output.questions
            if q in analysis.question_stats
        ]
        question_stds = [
            analysis.question_stats[q].std
            for q in output.questions
            if q in analysis.question_stats
        ]
        q_labels = [
            analysis.question_labels[i]
            for i, q in enumerate(output.questions)
            if q in analysis.question_stats
        ]

        y_pos = np.arange(len(q_labels))
        colors = [score_cmap(m) for m in question_means]

        ax_question_bars.barh(
            y_pos,
            question_means,
            xerr=question_stds,
            capsize=4,
            color=colors,
            alpha=0.9,
        )
        ax_question_bars.set_yticks(y_pos)
        ax_question_bars.set_yticklabels(q_labels, fontsize=8)
        ax_question_bars.set_xlim(0, 1)
        ax_question_bars.set_title("Question Comparison", fontsize=12, pad=10)

    # Summary
    if ax_summary is not None:
        ax_summary.axis("off")
        summary_text = f"Models: {n_models}\nQuestions: {n_questions}\nAgreement: {analysis.model_agreement:.3f}\nDiscrimination: {analysis.question_discrimination:.3f}"
        ax_summary.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment="center")

    string_preview = (
        output.string[:60] + "..." if len(output.string) > 60 else output.string
    )
    fig.suptitle(
        f'Judge Analysis: "{string_preview}"', fontsize=14, fontweight="bold", y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


# -----------------------------------------------------------------------------
# Core Logic
# -----------------------------------------------------------------------------


def judge_string(
    inp: JudgeStringInput,
    profile: bool = False,
    isolate: bool = False,
) -> JudgeStringOutput:
    """Main judging logic using JudgeStructure."""
    output = JudgeStringOutput(
        string=inp.string,
        questions=inp.questions,
    )
    profiles = [] if profile else None

    # Local models
    for i, model_name in enumerate(inp.models, 1):
        print(f"\n{'=' * 60}")
        if profile:
            print(f"[PROFILE {i}/{len(inp.models)}] Model: {model_name}")
        else:
            print(f"Model: {model_name}")
        print("=" * 60)

        model_results, load_time, inference_time = judge_model(
            model_name=model_name,
            string=inp.string,
            questions=inp.questions,
            use_cloud=False,
            isolate=isolate,
            verbose=True,
        )

        output.model_results[model_name] = model_results

        if profile:
            total_time = load_time + inference_time
            print(f"\n  LOAD: {load_time:.2f}s", flush=True)
            print(f"  INFERENCE: {inference_time:.2f}s", flush=True)
            print(f"  TOTAL: {total_time:.2f}s", flush=True)
            profiles.append(
                ModelProfile(model_name, load_time, inference_time, total_time)
            )

    # Cloud models
    if inp.cloud_models:
        for model_name in inp.cloud_models:
            print(f"\n{'=' * 60}")
            print(f"[CLOUD] Model: {model_name}")
            print("=" * 60)

            model_results, _, inference_time = judge_model(
                model_name=model_name,
                string=inp.string,
                questions=inp.questions,
                use_cloud=True,
                isolate=False,
                verbose=True,
            )

            output.model_results[model_name] = model_results

            if profile:
                print(f"\n  TOTAL: {inference_time:.2f}s", flush=True)
                profiles.append(
                    ModelProfile(model_name, 0.0, inference_time, inference_time)
                )

    output.analysis = compute_analysis(output.model_results, inp.questions)
    output.profiles = profiles

    return output


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG, help="Config JSON")
    parser.add_argument("--viz", action="store_true", help="Show visualization")
    parser.add_argument(
        "--save-viz", type=Path, default=None, help="Save visualization"
    )
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument("--isolate", action="store_true", help="Run in subprocess")
    return parser.parse_args()


def print_output(output: JudgeStringOutput) -> None:
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f'\nString: "{output.string}"')
    print(f"\nQuestions ({len(output.questions)}):")
    for i, q in enumerate(output.questions, 1):
        print(f"  {i}. {q}")

    print("\n" + "-" * 70)
    print("MODEL JUDGMENTS")
    print("-" * 70)

    for model_name, results in output.model_results.items():
        print(f"\n  {model_name.split('/')[-1]}:")
        for q, j in results.judgments.items():
            status = "\u2713" if j.success else "\u2717"
            score_str = (
                f"{j.parsed_score:.3f}" if j.parsed_score is not None else "FAILED"
            )
            q_short = q[:40] + "..." if len(q) > 40 else q
            print(f"    {status} [{score_str:>6}] {q_short}")

    print("\n" + "-" * 70)
    print("JSON OUTPUT")
    print("-" * 70)
    print(json.dumps(output.to_dict(), indent=2))


def print_profile_summary(output: JudgeStringOutput) -> None:
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

    path = Path(args.config)
    if not path.exists():
        print(f"Error: {path} not found")
        return 1

    inp = JudgeStringInput.from_json(
        path, visualize=args.viz or args.save_viz is not None
    )

    print("=" * 70)
    print("STRING JUDGMENT (using JudgeStructure)")
    print("=" * 70)
    print(f'\nString: "{inp.string}"')
    print(f"\nQuestions ({len(inp.questions)}):")
    for i, q in enumerate(inp.questions, 1):
        print(f"  {i}. {q}")
    print(f"\nModels (local): {inp.models}")
    if inp.cloud_models:
        print(f"Models (cloud): {inp.cloud_models}")
    if args.profile:
        print("Profiling: ENABLED")
    if args.isolate:
        print("Isolation: ENABLED")

    output = judge_string(inp, profile=args.profile, isolate=args.isolate)

    if args.profile:
        print_profile_summary(output)

    print_output(output)

    if args.viz or args.save_viz:
        create_visualizations(output, save_path=args.save_viz)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
