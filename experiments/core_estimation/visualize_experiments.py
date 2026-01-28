"""Visualize experiment results as trajectory trees."""

from __future__ import annotations

import json
from pathlib import Path

from plot import plot_tree
from trees import build_tree


def visualize_results(result_dir: Path, output_dir: Path) -> None:
    """Visualize all experiment results in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for gen_file in result_dir.glob("gen_*.json"):
        print(f"\nProcessing: {gen_file.name}")

        with open(gen_file) as f:
            data = json.load(f)

        trajectories = data["trajectories"]
        prompt = data.get("prompt_text", "<root>")
        variant = data["prompt_variant"]

        if not trajectories:
            print("  No trajectories")
            continue

        print(f"  {len(trajectories)} trajectories")

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
                print("  Skipping token tree (no token data)")


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
        print(f"  Loaded {len(structures)} structures")

    return scores, structures
