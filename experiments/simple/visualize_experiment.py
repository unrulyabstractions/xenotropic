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
from dataclasses import dataclass, field
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
    children: dict[str, TreeNode] = field(default_factory=dict)
    count: int = 1  # Number of trajectories through this node
    # Layout fields (set during layout computation)
    x: float = 0.0
    y: float = 0.0

    @classmethod
    def create_root(cls) -> TreeNode:
        return cls(label="<root>", probability=1.0, cumulative_prob=1.0)


# -----------------------------------------------------------------------------
# Tree Building
# -----------------------------------------------------------------------------


def has_token_data(trajectories: list[dict]) -> bool:
    """Check if trajectories have actual token-level data."""
    for traj in trajectories:
        if traj.get("per_token_logprobs"):
            return True
    return False


def build_token_tree(trajectories: list[dict]) -> TreeNode:
    """Build a tree from token sequences. Returns None if no token data."""
    # Check if we have actual token data
    if not has_token_data(trajectories):
        return None

    root = TreeNode.create_root()

    for traj in trajectories:
        prob = traj["probability"]
        tokens = traj.get("per_token_logprobs", [])

        if not tokens:
            continue

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
                    count=0,
                )

            current.children[chunk].count += 1
            current = current.children[chunk]

    return root


# -----------------------------------------------------------------------------
# Tree Layout Algorithm
# -----------------------------------------------------------------------------


def compute_layout(
    node: TreeNode,
    depth: int = 0,
    y_offset: float = 0,
    x_spacing: float = 1.0,
    y_spacing: float = 1.0,
) -> float:
    """
    Compute tree layout using a simple algorithm.

    - X coordinate = depth (horizontal: root left, leaves right)
    - Y coordinate = computed to avoid overlaps

    Returns the total height used by this subtree.
    """
    node.x = depth * x_spacing

    if not node.children:
        # Leaf node
        node.y = y_offset
        return 1.0  # Height of 1 unit

    # Sort children by count (most frequent first, at top)
    sorted_children = sorted(node.children.values(), key=lambda c: -c.count)

    # Layout children
    current_y = y_offset
    total_height = 0
    child_positions = []

    for child in sorted_children:
        child_height = compute_layout(
            child,
            depth + 1,
            current_y,
            x_spacing,
            y_spacing,
        )
        child_positions.append(child.y)
        current_y += child_height * y_spacing
        total_height += child_height * y_spacing

    # Center parent at middle of children
    if child_positions:
        node.y = (child_positions[0] + child_positions[-1]) / 2
    else:
        node.y = y_offset

    return max(total_height / y_spacing, 1.0)


def count_leaves(node: TreeNode) -> int:
    """Count leaf nodes in subtree."""
    if not node.children:
        return 1
    return sum(count_leaves(c) for c in node.children.values())


def get_max_depth(node: TreeNode, current: int = 0) -> int:
    """Get maximum depth of tree."""
    if not node.children:
        return current
    return max(get_max_depth(c, current + 1) for c in node.children.values())


def prune_tree(
    node: TreeNode, max_depth: int, min_count: int, depth: int = 0
) -> Optional[TreeNode]:
    """Prune tree to max_depth and min_count."""
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
            pruned = prune_tree(child, max_depth, min_count, depth + 1)
            if pruned:
                pruned_children[label] = pruned

    return TreeNode(
        label=node.label,
        probability=node.probability,
        cumulative_prob=node.cumulative_prob,
        children=pruned_children,
        count=node.count,
    )


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------


def collect_tree_data(
    node: TreeNode, parent_pos: Optional[tuple] = None
) -> tuple[list, list]:
    """Collect all edges and nodes for plotting."""
    edges = []  # (start_pos, end_pos, probability)
    nodes = []  # (pos, label, probability, count)

    pos = (node.x, node.y)
    nodes.append((pos, node.label, node.probability, node.count))

    if parent_pos is not None:
        edges.append((parent_pos, pos, node.probability))

    for child in node.children.values():
        child_edges, child_nodes = collect_tree_data(child, pos)
        edges.extend(child_edges)
        nodes.extend(child_nodes)

    return edges, nodes


def plot_tree(
    root: TreeNode,
    title: str,
    output_path: Path,
    max_depth: int = 5,
    min_count: int = 1,
) -> None:
    """Plot a trajectory tree with horizontal layout."""

    # Prune tree
    pruned = prune_tree(root, max_depth, min_count)
    if not pruned:
        print(f"  No nodes to plot for {title}")
        return

    # Count nodes for sizing
    n_leaves = count_leaves(pruned)
    tree_depth = get_max_depth(pruned)

    # Compute spacing based on tree size
    x_spacing = 2.0  # Horizontal spacing between depths
    y_spacing = 1.0  # Vertical spacing between siblings

    # Compute layout
    compute_layout(pruned, x_spacing=x_spacing, y_spacing=y_spacing)

    # Collect data
    edges, nodes = collect_tree_data(pruned)

    if not nodes:
        print(f"  No nodes to plot for {title}")
        return

    # Compute figure size based on tree dimensions
    fig_width = max(10, (tree_depth + 1) * 2.5)
    fig_height = max(6, n_leaves * 0.6)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Get coordinate ranges for proper sizing
    all_x = [n[0][0] for n in nodes]
    all_y = [n[0][1] for n in nodes]
    x_range = max(all_x) - min(all_x) if len(set(all_x)) > 1 else 1
    y_range = max(all_y) - min(all_y) if len(set(all_y)) > 1 else 1

    # Draw edges with probability labels
    for start, end, prob in edges:
        # Color by probability
        color = plt.cm.RdYlGn(prob)
        linewidth = 0.8 + prob * 2.5

        # Draw edge
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=linewidth,
            alpha=0.8,
            zorder=1,
        )

        # Add probability label on edge
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2

        # Format probability nicely
        if prob >= 0.01:
            prob_text = f"{prob:.2f}"
        else:
            prob_text = f"{prob:.1e}"

        # Add white background for readability
        ax.annotate(
            prob_text,
            (mid_x, mid_y),
            fontsize=7,
            ha="center",
            va="center",
            color="black",
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="none",
                alpha=0.8,
            ),
            zorder=3,
        )

    # Draw nodes
    for pos, label, prob, count in nodes:
        # Size by count
        size = 80 + count * 40
        color = plt.cm.RdYlGn(prob)

        ax.scatter(
            [pos[0]],
            [pos[1]],
            s=size,
            c=[color],
            alpha=0.9,
            edgecolors="black",
            linewidths=1,
            zorder=2,
        )

        # Label (truncate if too long)
        max_label_len = 20
        display_label = (
            label[:max_label_len] + "..." if len(label) > max_label_len else label
        )
        display_label = display_label.replace("\n", "\\n")

        # Position label to the right of node
        ax.annotate(
            display_label,
            (pos[0] + 0.15, pos[1]),
            fontsize=8,
            ha="left",
            va="center",
            fontfamily="monospace",
            zorder=4,
        )

    # Title
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    # Clean up axes
    ax.set_xlabel("Depth (token position)", fontsize=10)
    ax.axis("off")

    # Add margins
    x_margin = x_range * 0.15 + 1
    y_margin = max(y_range * 0.1, 0.5)
    ax.set_xlim(min(all_x) - 0.5, max(all_x) + x_margin + 2)
    ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)

    # Add colorbar legend
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("P(token | prefix)", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
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

        # Build and plot word tree (most useful view)
        word_tree = build_word_tree(trajectories)
        plot_tree(
            word_tree,
            f"Word Tree - {prompt_variant}",
            output_dir / f"{prompt_variant}_word_tree.png",
            max_depth=10,
            min_count=1,
        )

        # Build and plot sentence tree
        sentence_tree = build_sentence_tree(trajectories, chunk_size=3)
        plot_tree(
            sentence_tree,
            f"Phrase Tree - {prompt_variant}",
            output_dir / f"{prompt_variant}_phrase_tree.png",
            max_depth=6,
            min_count=1,
        )

        # Build and plot token tree (only if we have token data)
        token_tree = build_token_tree(trajectories)
        if token_tree is not None:
            plot_tree(
                token_tree,
                f"Token Tree - {prompt_variant}",
                output_dir / f"{prompt_variant}_token_tree.png",
                max_depth=12,
                min_count=1,
            )
        else:
            print("  Skipping token tree (no per-token data available)")


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
