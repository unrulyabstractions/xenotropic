"""
Artificial Hivemind - Explore multi-agent conversation datasets.

Uses data from: https://huggingface.co/collections/liweijiang/artificial-hivemind
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from datasets import load_dataset

# -----------------------------------------------------------------------------
# Input/Output Data Structures
# -----------------------------------------------------------------------------


@dataclass
class HivemindInput:
    """Input for hivemind exploration."""

    datasets: list[str]


@dataclass
class DatasetInfo:
    """Info about a single dataset."""

    name: str
    keys: list[str]
    first_example: dict


@dataclass
class HivemindOutput:
    """Output from hivemind exploration."""

    dataset_infos: list[DatasetInfo]


# -----------------------------------------------------------------------------
# Core Logic
# -----------------------------------------------------------------------------


def explore_dataset(dataset_name: str) -> DatasetInfo:
    """Explore the structure of a single dataset."""
    dataset = load_dataset(dataset_name, split="train")
    example = dataset[0]

    return DatasetInfo(
        name=dataset_name,
        keys=list(example.keys()),
        first_example=example,
    )


def explore_hivemind(inp: HivemindInput) -> HivemindOutput:
    """Explore all specified datasets."""
    infos = []
    for dataset_name in inp.datasets:
        infos.append(explore_dataset(dataset_name))
    return HivemindOutput(dataset_infos=infos)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=[
            "liweijiang/infinite-chats-taxonomy",
            "liweijiang/infinite-chats-eval",
            "liweijiang/infinite-chats-human-absolute",
            "liweijiang/infinite-chats-human-pairwise",
        ],
        help="Dataset names to explore",
    )
    return parser.parse_args()


def input_from_args(args: argparse.Namespace) -> HivemindInput:
    """Load input from command line arguments."""
    return HivemindInput(datasets=args.datasets)


def save_output(args: argparse.Namespace, output: HivemindOutput) -> None:
    """Save output (no-op for this script)."""
    pass


def print_output(args: argparse.Namespace, output: HivemindOutput) -> None:
    """Print output to stdout."""
    for info in output.dataset_infos:
        print("=" * 60)
        print(f"\n\nDataset Structure: {info.name}")
        print("=" * 60)

        print(f"\nKeys: {info.keys}\n")

        for key, value in info.first_example.items():
            print(f"{key}:")
            if isinstance(value, (list, dict)):
                print(f"  Type: {type(value).__name__}")
                if isinstance(value, list) and len(value) > 0:
                    print(f"  Length: {len(value)}")
                    print(f"  First item: {value[0]}")
            else:
                print(f"  {value}")
            print()


def main() -> int:
    args = get_args()
    inp: HivemindInput = input_from_args(args)
    output: HivemindOutput = explore_hivemind(inp)

    save_output(args, output)
    print_output(args, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
