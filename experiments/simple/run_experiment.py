#!/usr/bin/env python3
"""
Run experiment: collect trajectories, evaluate with judge, estimate cores.

Usage: python run_experiment.py [trial] [--no-viz]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiment import Experiment

SCRIPT_DIR = Path(__file__).parent


def get_args():
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("trial", nargs="?", default="test")
    parser.add_argument("--target-mass", type=float, default=0.9)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = get_args()

    exp = Experiment.from_trial(args.trial, synthetic=False)
    exp.run(
        target_mass=args.target_mass,
        max_iterations=args.max_iterations,
        verbose=not args.quiet,
    )
    exp.save()

    if not args.no_viz:
        exp.visualize()

    exp.print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
