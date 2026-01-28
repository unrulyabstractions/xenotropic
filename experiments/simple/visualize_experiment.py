#!/usr/bin/env python3
"""
Visualize experiment results as trajectory trees.

Creates visualizations of trajectories as:
1. Token tree - each node is a token, edges show conditional probabilities
2. Word tree - each node is a word, edges show conditional probabilities
3. Sentence tree - each node is a sentence fragment

Usage: python visualize_experiment.py [trial_name]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------


@dataclass
class TreeNode:
    """Node in a trajectory tree."""

    label: str
    probability: float  # Conditional probability P(this | parent)
    cumulative_prob: float  # P(root -> this)
    children: dict[str, TreeNode]  # label -> child node
    count: int = 1  # Number of trajectories through this node

    @classmethod
    def create_root(cls) -> TreeNode:
        return cls(label="<root>", probability=1.0, cumulative_prob=1.0, children={})


# -----------------------------------------------------------------------------
# Tree Building
# -----------------------------------------------------------------------------


def build_token_tree(trajectories: list[dict]) -> TreeNode:
    """Build a tree from token sequences."""
    root = TreeNode.create_root()

    for traj in trajectories:
        prob = traj["probability"]
        tokens = traj.get("per_token_logprobs", [])

        if not tokens:
            # Fall back to splitting text if no token info
            text = traj["text"]
            tokens = [{"token": c, "logprob": 0} for c in text]

        current = root
        cumulative = 1.0

        for tok_info in tokens:
            token = tok_info["token"]
            logprob = tok_info.get("logprob", 0)
            cond_prob = np.exp(logprob) if logprob else prob ** (1 / len(tokens))

            cumulative *= cond_prob

            if token not in current.children:
                current.children[token] = TreeNode(
                    label=token,
                    probability=cond_prob,
                    cumulative_prob=cumulative,
                    children={},
                    count=0,
                )

            current.children[token].count += 1
            current = current.children[token]

    return root


def build_word_tree(trajectories: list[dict]) -> TreeNode:
    """Build a tree from word sequences."""
    root = TreeNode.create_root()

    for traj in trajectories:
        prob = traj["probability"]
        text = traj["text"]

        # Split into words (keep punctuation attached)
        words = re.findall(r"\S+", text)
        if not words:
            continue

        current = root
        cumulative = 1.0
        # Distribute probability across words
        word_prob = prob ** (1 / len(words)) if words else 1.0

        for word in words:
            cumulative *= word_prob

            if word not in current.children:
                current.children[word] = TreeNode(
                    label=word,
                    probability=word_prob,
                    cumulative_prob=cumulative,
                    children={},
                    count=0,
                )

            current.children[word].count += 1
            current = current.children[word]

    return root


def build_sentence_tree(trajectories: list[dict], chunk_size: int = 3) -> TreeNode:
    """Build a tree from sentence/phrase chunks."""
    root = TreeNode.create_root()

    for traj in trajectories:
        prob = traj["probability"]
        text = traj["text"]

        # Split into words then group into chunks
        words = re.findall(r"\S+", text)
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            chunks.append(chunk)

        if not chunks:
            continue

        current = root
        cumulative = 1.0
        chunk_prob = prob ** (1 / len(chunks)) if chunks else 1.0

        for chunk in chunks:
            cumulative *= chunk_prob

            if chunk not in current.children:
                current.children[chunk] = TreeNode(
                    label=chunk,
                    probability=chunk_prob,
                    cumulative_prob=cumulative,
                    children={},
                    count=0,
                )

            current.children[chunk].count += 1
            current = current.children[chunk]

    return root


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------


def collect_edges(
    node: TreeNode,
    parent_pos: Optional[tuple] = None,
    depth: int = 0,
    pos_x: float = 0,
    width: float = 1.0,
) -> tuple[list, list, list]:
    """Collect edges, nodes, and labels for plotting."""
    edges = []
    nodes = []
    labels = []

    pos = (pos_x, -depth)
    nodes.append((pos, node.label, node.probability, node.count))

    if parent_pos is not None:
        edges.append((parent_pos, pos, node.probability))

    if node.children:
        n_children = len(node.children)
        child_width = width / max(n_children, 1)

        for i, (label, child) in enumerate(
            sorted(node.children.items(), key=lambda x: -x[1].count)
        ):
            child_x = pos_x - width / 2 + child_width * (i + 0.5)
            child_edges, child_nodes, child_labels = collect_edges(
                child, pos, depth + 1, child_x, child_width
            )
            edges.extend(child_edges)
            nodes.extend(child_nodes)
            labels.extend(child_labels)

    return edges, nodes, labels


def plot_tree(
    root: TreeNode,
    title: str,
    output_path: Path,
    max_depth: int = 5,
    min_count: int = 1,
) -> None:
    """Plot a trajectory tree."""

    # Prune tree to max_depth and min_count
    def prune(node: TreeNode, depth: int) -> TreeNode:
        if depth >= max_depth:
            return TreeNode(
                label=node.label,
                probability=node.probability,
                cumulative_prob=node.cumulative_prob,
                children={},
                count=node.count,
            )

        pruned_children = {}
        for label, child in node.children.items():
            if child.count >= min_count:
                pruned_children[label] = prune(child, depth + 1)

        return TreeNode(
            label=node.label,
            probability=node.probability,
            cumulative_prob=node.cumulative_prob,
            children=pruned_children,
            count=node.count,
        )

    pruned = prune(root, 0)
    edges, nodes, _ = collect_edges(pruned, width=10.0)

    if not nodes:
        print(f"  No nodes to plot for {title}")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Draw edges
    for start, end, prob in edges:
        # Color by probability (green=high, red=low)
        color = plt.cm.RdYlGn(prob)
        linewidth = 0.5 + prob * 2
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=linewidth,
            alpha=0.7,
        )

    # Draw nodes
    for pos, label, prob, count in nodes:
        # Size by count
        size = 100 + count * 50
        color = plt.cm.RdYlGn(prob)

        ax.scatter([pos[0]], [pos[1]], s=size, c=[color], alpha=0.8, edgecolors="black")

        # Label (truncate if too long)
        display_label = label[:15] + "..." if len(label) > 15 else label
        display_label = display_label.replace("\n", "\\n")
        ax.annotate(
            display_label,
            pos,
            fontsize=7,
            ha="center",
            va="bottom",
            rotation=45,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Branches")
    ax.set_ylabel("Depth")
    ax.set_aspect("equal")
    ax.axis("off")

    # Add colorbar legend
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
    cbar.set_label("Conditional Probability")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_results(result_dir: Path, output_dir: Path) -> None:
    """Visualize all results in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find generation files
    gen_files = list(result_dir.glob("gen_*.json"))
    if not gen_files:
        print(f"No generation files found in {result_dir}")
        return

    for gen_file in gen_files:
        print(f"\nProcessing: {gen_file.name}")

        with open(gen_file) as f:
            gen_data = json.load(f)

        trajectories = gen_data["trajectories"]
        prompt_variant = gen_data["prompt_variant"]

        if not trajectories:
            print("  No trajectories to visualize")
            continue

        print(f"  {len(trajectories)} trajectories")

        # Build and plot token tree
        token_tree = build_token_tree(trajectories)
        plot_tree(
            token_tree,
            f"Token Tree - {prompt_variant}",
            output_dir / f"{prompt_variant}_token_tree.png",
            max_depth=8,
            min_count=1,
        )

        # Build and plot word tree
        word_tree = build_word_tree(trajectories)
        plot_tree(
            word_tree,
            f"Word Tree - {prompt_variant}",
            output_dir / f"{prompt_variant}_word_tree.png",
            max_depth=6,
            min_count=1,
        )

        # Build and plot sentence tree
        sentence_tree = build_sentence_tree(trajectories, chunk_size=3)
        plot_tree(
            sentence_tree,
            f"Sentence Tree - {prompt_variant}",
            output_dir / f"{prompt_variant}_sentence_tree.png",
            max_depth=4,
            min_count=1,
        )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "trial",
        nargs="?",
        default="test",
        help="Trial name (folder in out/)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Visualize synthetic results (adds _synthetic suffix)",
    )
    return parser.parse_args()


def main() -> int:
    args = get_args()

    # Determine result directory
    trial_name = args.trial
    if args.synthetic:
        trial_name = f"{trial_name}_synthetic"

    result_dir = Path(__file__).parent / "out" / trial_name

    if not result_dir.exists():
        print(f"Result directory not found: {result_dir}")
        print("\nAvailable results:")
        out_dir = Path(__file__).parent / "out"
        if out_dir.exists():
            for d in out_dir.iterdir():
                if d.is_dir():
                    print(f"  {d.name}")
        return 1

    # Output directory
    viz_dir = result_dir / "viz"

    print(f"Visualizing: {result_dir}")
    print(f"Output: {viz_dir}")

    visualize_results(result_dir, viz_dir)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
