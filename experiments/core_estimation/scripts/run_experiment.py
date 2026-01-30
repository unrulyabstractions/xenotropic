#!/usr/bin/env python3
"""Run experiment: collect trajectories, evaluate with judge, estimate cores.

Usage: python run_experiment.py [trial] [--no-viz] [--max-trajectories N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
sys.path.insert(0, str(SCRIPT_DIR.parent.parent.parent))

from experiment import Experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("trial", nargs="?", default="default")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Override max trajectories per branch point (default: from trial config)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override max new tokens per trajectory (default: from trial config)",
    )
    parser.add_argument(
        "--max-viz-samples-per-branch-point",
        type=int,
        default=5,
        help="Max trajectories to visualize per branch point (top by probability)",
    )
    args = parser.parse_args()

    exp = Experiment.from_trial(args.trial)
    exp.run(
        max_trajectories=args.max_trajectories,
        max_new_tokens=args.max_new_tokens,
    )
    exp.save()

    if not args.no_viz:
        exp.visualize(
            max_viz_samples_per_branch_point=args.max_viz_samples_per_branch_point
        )

    exp.print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
