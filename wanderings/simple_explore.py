"""
Simple exploration script using Explorer abstraction.

Runs greedy generation and builds TreeNode.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from exploration import GreedyGenerator
from xenotechnics.common import String

# -----------------------------------------------------------------------------
# Input/Output Data Structures
# -----------------------------------------------------------------------------


@dataclass
class SimpleExploreInput:
    """Input for simple exploration."""

    prompt: str
    model_name: str
    max_new_tokens: int


@dataclass
class SimpleExploreOutput:
    """Output from simple exploration."""

    tree_depth: int
    num_trajectories: int
    trajectories: list[dict]


# -----------------------------------------------------------------------------
# Core Logic
# -----------------------------------------------------------------------------


def simple_explore(inp: SimpleExploreInput) -> SimpleExploreOutput:
    """Run greedy exploration and return results."""
    prompt = String.from_text(inp.prompt)

    print("=" * 60)
    print("GREEDY EXPLORATION")
    print("=" * 60)
    print()

    explorer = GreedyGenerator(model_name=inp.model_name)
    tree = explorer.run(prompt=prompt, max_new_tokens=inp.max_new_tokens, verbose=True)

    trajectory_nodes = tree.get_trajectory_nodes()
    trajectories = []

    for i, trajectory_node in enumerate(trajectory_nodes):
        cont_logprob = trajectory_node.get_continuation_logprob(
            explorer.prompt_token_count
        )
        cont_prob = trajectory_node.get_continuation_prob(explorer.prompt_token_count)
        trajectories.append(
            {
                "index": i,
                "text": trajectory_node.string.to_text(),
                "prob": cont_prob,
                "logprob": cont_logprob,
            }
        )

    return SimpleExploreOutput(
        tree_depth=tree.depth(),
        num_trajectories=len(trajectory_nodes),
        trajectories=trajectories,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--prompt",
        type=str,
        default="Complete the following sentence in less than 3 words. Just give me completion: Roses are",
        help="Prompt to complete",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-2b-it",
        help="Model name",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum new tokens to generate",
    )
    return parser.parse_args()


def input_from_args(args: argparse.Namespace) -> SimpleExploreInput:
    """Load input from command line arguments."""
    return SimpleExploreInput(
        prompt=args.prompt,
        model_name=args.model,
        max_new_tokens=args.max_tokens,
    )


def save_output(args: argparse.Namespace, output: SimpleExploreOutput) -> None:
    """Save output (no-op for this script)."""
    pass


def print_output(args: argparse.Namespace, output: SimpleExploreOutput) -> None:
    """Print output to stdout."""
    print("\n" + "=" * 60)
    print("TreeNode Analysis:")
    print("=" * 60)
    print(f"Root depth: {output.tree_depth}")
    print(f"Number of trajectories: {output.num_trajectories}")

    for traj in output.trajectories:
        print(f"\nTrajectory {traj['index']}")
        print(f"  Continuation prob={traj['prob']:.6e} | logprob={traj['logprob']:.4f}")


def main() -> int:
    args = get_args()
    inp: SimpleExploreInput = input_from_args(args)
    output: SimpleExploreOutput = simple_explore(inp)

    save_output(args, output)
    print_output(args, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
