#!/usr/bin/env python3
"""
Run experiment: collect trajectories, evaluate with judge, estimate cores.

Usage: python run_experiment.py [trial] [--no-viz] [--quiet]

Set model to "synthetic" in trial config for synthetic mode.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiment import Experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("trial", nargs="?", default="test")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    args = parser.parse_args()

    exp = Experiment.from_trial(args.trial)
    exp.run(verbose=not args.quiet)
    exp.save()

    if not args.no_viz:
        exp.visualize()

    exp.print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
