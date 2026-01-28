#!/usr/bin/env python3
"""Visualize experiment results as trajectory trees.

Usage: python visualize_experiments.py [experiment_name]

If no experiment name given, visualizes all experiments in out/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from plot import plot_tree
from trees import build_tree

SCRIPT_DIR = Path(__file__).parent
OUT_DIR = SCRIPT_DIR / "out"


def visualize_experiment(result_dir: Path, output_dir: Path | None = None) -> None:
    """Visualize a single experiment's results."""
    if output_dir is None:
        output_dir = result_dir / "viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    for gen_file in result_dir.glob("gen_*.json"):
        print(f"  {gen_file.name}")

        with open(gen_file) as f:
            data = json.load(f)

        trajectories = data["trajectories"]
        prompt = data.get("prompt_text", "<root>")
        variant = data["prompt_variant"]

        if not trajectories:
            print("    No trajectories")
            continue

        scores, structures = _load_scores(result_dir, variant, trajectories)

        for mode in ["word", "phrase", "token"]:
            tree = build_tree(trajectories, scores, prompt, mode)
            if tree:
                plot_tree(
                    tree,
                    f"{mode.title()} Tree",
                    output_dir / f"{mode}_tree.png",
                    structures,
                    scores,
                )
            elif mode == "token":
                print("    Skipping token tree (no token data)")


def visualize_all() -> None:
    """Visualize all experiments in out/."""
    if not OUT_DIR.exists():
        print(f"No experiments found in {OUT_DIR}")
        return

    experiments = [
        d for d in OUT_DIR.iterdir() if d.is_dir() and list(d.glob("gen_*.json"))
    ]

    if not experiments:
        print(f"No experiments with results found in {OUT_DIR}")
        return

    print(f"Found {len(experiments)} experiment(s)\n")

    for exp_dir in sorted(experiments):
        print(f"\n{exp_dir.name}/")
        visualize_experiment(exp_dir)


def _load_scores(
    result_dir: Path, variant: str, trajectories: list[dict]
) -> tuple[dict[str, list[float]], list[str]]:
    """Load structure scores from estimation file."""
    scores, structures = {}, []
    est_file = result_dir / f"est_{variant}.json"

    if est_file.exists():
        with open(est_file) as f:
            est = json.load(f)
        if est.get("systems"):
            sys_data = est["systems"][0]
            structures = [s["structure"] for s in sys_data["structures"]]
            for i, t in enumerate(trajectories):
                scores[t["text"]] = [s["scores"][i] for s in sys_data["structures"]]
        print(f"    {len(structures)} structures")

    return scores, structures


def main() -> int:
    if len(sys.argv) > 1:
        name = sys.argv[1]
        exp_dir = OUT_DIR / name
        if not exp_dir.exists():
            print(f"Experiment '{name}' not found in {OUT_DIR}")
            return 1
        print(f"{name}/")
        visualize_experiment(exp_dir)
    else:
        visualize_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
