#!/usr/bin/env python3
"""
Run experiment with synthetic data for testing/development.

Usage: python run_estimation_with_synthetic.py [trial] [--no-viz]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiment import Experiment


def get_args():
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("trial", nargs="?", default="test")
    parser.add_argument("--num-trajectories", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = get_args()

    exp = Experiment.from_trial(args.trial, synthetic=True)
    exp.run(
        num_trajectories=args.num_trajectories,
        seed=args.seed,
        verbose=not args.quiet,
    )
    exp.save()

    if not args.no_viz:
        exp.visualize()

    exp.print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
