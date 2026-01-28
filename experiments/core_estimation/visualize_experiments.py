#!/usr/bin/env python3
"""Visualize experiment results as trajectory trees.

Usage: python visualize_experiments.py [experiment_name]

If no experiment name given, visualizes all experiments in out/.
"""

from __future__ import annotations

import sys
from pathlib import Path

from plot import visualize_experiment

SCRIPT_DIR = Path(__file__).parent
OUT_DIR = SCRIPT_DIR / "out"


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


if __name__ == "__main__":
    raise SystemExit(main())
