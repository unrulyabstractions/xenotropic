#!/usr/bin/env python3
"""
Visualize experiment results as trajectory trees.

Creates visualizations of trajectories as:
1. Word tree - each node is a word, edges show relative probabilities
2. Phrase tree - each node is a phrase chunk

Leaf nodes are colored by dominant structure compliance.
Edge thickness is proportional to relative probability at branch point.

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

# Pastel colors for structures
PASTEL_COLORS = [
    "#FFB3BA",  # pastel red/pink
    "#BAFFC9",  # pastel green
    "#BAE1FF",  # pastel blue
    "#FFFFBA",  # pastel yellow
    "#FFDFba",  # pastel orange
    "#E0BBE4",  # pastel purple
]


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------


@dataclass
class TreeNode:
    """Node in a trajectory tree."""

    label: str
    probability: float  # Absolute probability of this trajectory prefix
    children: dict[str, TreeNode] = field(default_factory=dict)
    count: int = 1  # Number of trajectories through this node
    # Layout fields (set during layout computation)
    x: float = 0.0
    y: float = 0.0
    # Structure compliance (only for leaf nodes)
    structure_scores: Optional[list[float]] = None
    trajectory_text: Optional[str] = None  # Full trajectory text for leaf nodes

    @classmethod
    def create_root(cls) -> TreeNode:
        return cls(label="<root>", probability=1.0)

    def is_leaf(self) -> bool:
        return len(self.children) == 0


# -----------------------------------------------------------------------------
# Tree Building
# -----------------------------------------------------------------------------


def build_word_tree(
    trajectories: list[dict],
    trajectory_scores: Optional[dict[str, list[float]]] = None,
) -> TreeNode:
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
        # Distribute probability across words
        word_prob = prob ** (1 / len(words)) if words else 1.0

        for i, word in enumerate(words):
            is_last = i == len(words) - 1

            if word not in current.children:
                current.children[word] = TreeNode(
                    label=word,
                    probability=word_prob,
                    count=0,
                )

            current.children[word].count += 1
            current.children[word].probability = word_prob

            # If this is a leaf node, store structure scores
            if is_last and trajectory_scores and text in trajectory_scores:
                current.children[word].structure_scores = trajectory_scores[text]
                current.children[word].trajectory_text = text

            current = current.children[word]

    return root


def build_phrase_tree(
    trajectories: list[dict],
    trajectory_scores: Optional[dict[str, list[float]]] = None,
    chunk_size: int = 3,
) -> TreeNode:
    """Build a tree from phrase chunks."""
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
        chunk_prob = prob ** (1 / len(chunks)) if chunks else 1.0

        for i, chunk in enumerate(chunks):
            is_last = i == len(chunks) - 1

            if chunk not in current.children:
                current.children[chunk] = TreeNode(
                    label=chunk,
                    probability=chunk_prob,
                    count=0,
                )

            current.children[chunk].count += 1
            current.children[chunk].probability = chunk_prob

            # If this is a leaf node, store structure scores
            if is_last and trajectory_scores and text in trajectory_scores:
                current.children[chunk].structure_scores = trajectory_scores[text]
                current.children[chunk].trajectory_text = text

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
    Compute tree layout.
    Returns the total height used by this subtree.
    """
    node.x = depth * x_spacing

    if not node.children:
        node.y = y_offset
        return 1.0

    # Sort children by count (most frequent first, at top)
    sorted_children = sorted(node.children.values(), key=lambda c: -c.count)

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
            children={},
            count=node.count,
            structure_scores=node.structure_scores,
            trajectory_text=node.trajectory_text,
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
        children=pruned_children,
        count=node.count,
        structure_scores=node.structure_scores,
        trajectory_text=node.trajectory_text,
    )


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------


def collect_tree_data(
    node: TreeNode, parent_pos: Optional[tuple] = None
) -> tuple[list, list]:
    """Collect all edges and nodes for plotting."""
    edges = []  # (start_pos, end_pos, child_node, sibling_probs)
    nodes = []  # (node, pos)

    pos = (node.x, node.y)
    nodes.append((node, pos))

    if parent_pos is not None and node.children:
        # Nothing special for internal edges
        pass

    # Collect sibling probabilities for relative edge thickness
    if node.children:
        total_prob = sum(c.probability for c in node.children.values())
        for child in node.children.values():
            child_edges, child_nodes = collect_tree_data(child, pos)
            # Add edge with relative probability info
            edges.append((pos, (child.x, child.y), child, total_prob))
            edges.extend(child_edges)
            nodes.extend(child_nodes)

    return edges, nodes


def plot_tree(
    root: TreeNode,
    title: str,
    output_path: Path,
    structures: list[str],
    max_depth: int = 10,
    min_count: int = 1,
) -> None:
    """Plot a trajectory tree with structure-based coloring."""

    # Prune tree
    pruned = prune_tree(root, max_depth, min_count)
    if not pruned:
        print(f"  No nodes to plot for {title}")
        return

    n_leaves = count_leaves(pruned)
    tree_depth = get_max_depth(pruned)

    x_spacing = 2.5
    y_spacing = 1.2

    compute_layout(pruned, x_spacing=x_spacing, y_spacing=y_spacing)

    edges, nodes = collect_tree_data(pruned)

    if not nodes:
        print(f"  No nodes to plot for {title}")
        return

    # Figure sizing
    fig_width = max(12, (tree_depth + 1) * 3)
    fig_height = max(6, n_leaves * 0.8)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Get coordinate ranges
    all_x = [n[1][0] for n in nodes]
    all_y = [n[1][1] for n in nodes]
    x_range = max(all_x) - min(all_x) if len(set(all_x)) > 1 else 1
    y_range = max(all_y) - min(all_y) if len(set(all_y)) > 1 else 1

    # Draw edges with thickness proportional to relative probability
    for start, end, child_node, total_prob in edges:
        rel_prob = child_node.probability / total_prob if total_prob > 0 else 0.5
        linewidth = 0.5 + rel_prob * 4  # Scale thickness

        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="#666666",
            linewidth=linewidth,
            alpha=0.6,
            zorder=1,
            solid_capstyle="round",
        )

        # Add relative probability label
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2

        prob_text = f"{rel_prob:.2f}"
        ax.annotate(
            prob_text,
            (mid_x, mid_y),
            fontsize=7,
            ha="center",
            va="center",
            color="#333333",
            bbox=dict(
                boxstyle="round,pad=0.1",
                facecolor="white",
                edgecolor="none",
                alpha=0.7,
            ),
            zorder=3,
        )

    # Draw nodes
    for node, pos in nodes:
        is_leaf = node.is_leaf()

        if is_leaf and node.structure_scores is not None:
            # Color by dominant structure
            dominant_idx = int(np.argmax(node.structure_scores))
            node_color = PASTEL_COLORS[dominant_idx % len(PASTEL_COLORS)]
        else:
            node_color = "#DDDDDD"  # Gray for non-leaf nodes

        size = 60 + node.count * 30

        ax.scatter(
            [pos[0]],
            [pos[1]],
            s=size,
            c=[node_color],
            alpha=0.9,
            edgecolors="black",
            linewidths=0.8,
            zorder=2,
        )

        # Node label
        max_label_len = 18
        display_label = (
            node.label[:max_label_len] + "..."
            if len(node.label) > max_label_len
            else node.label
        )
        display_label = display_label.replace("\n", "\\n")

        label_offset = 0.15

        # For leaf nodes with structure scores, show colored compliance values
        if is_leaf and node.structure_scores is not None:
            # Build colored score text
            ax.annotate(
                display_label,
                (pos[0] + label_offset, pos[1]),
                fontsize=8,
                ha="left",
                va="center",
                fontfamily="monospace",
                zorder=4,
            )

            # Add structure scores as colored text
            score_x = pos[0] + label_offset + len(display_label) * 0.08 + 0.3
            score_text = "["
            ax.annotate(
                score_text,
                (score_x, pos[1]),
                fontsize=7,
                ha="left",
                va="center",
                fontfamily="monospace",
                zorder=4,
            )

            bracket_width = 0.08
            score_x += bracket_width

            for i, score in enumerate(node.structure_scores):
                color = PASTEL_COLORS[i % len(PASTEL_COLORS)]
                # Darken for text readability
                darker = tuple(int(c * 0.6) for c in bytes.fromhex(color[1:]))
                text_color = f"#{darker[0]:02x}{darker[1]:02x}{darker[2]:02x}"

                score_str = f"{score:.2f}"
                if i < len(node.structure_scores) - 1:
                    score_str += ", "

                ax.annotate(
                    score_str,
                    (score_x, pos[1]),
                    fontsize=7,
                    ha="left",
                    va="center",
                    fontfamily="monospace",
                    color=text_color,
                    fontweight="bold",
                    zorder=4,
                )
                score_x += len(score_str) * 0.065

            ax.annotate(
                "]",
                (score_x, pos[1]),
                fontsize=7,
                ha="left",
                va="center",
                fontfamily="monospace",
                zorder=4,
            )
        else:
            ax.annotate(
                display_label,
                (pos[0] + label_offset, pos[1]),
                fontsize=8,
                ha="left",
                va="center",
                fontfamily="monospace",
                zorder=4,
            )

    # Title
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")

    # Add margins
    x_margin = x_range * 0.15 + 3
    y_margin = max(y_range * 0.1, 0.5)
    ax.set_xlim(min(all_x) - 0.5, max(all_x) + x_margin)
    ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)

    # Add legend for structures
    if structures:
        legend_handles = []
        for i, struct in enumerate(structures):
            color = PASTEL_COLORS[i % len(PASTEL_COLORS)]
            # Truncate structure name for legend
            short_name = struct[:30] + "..." if len(struct) > 30 else struct
            patch = plt.Rectangle(
                (0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.5
            )
            legend_handles.append((patch, short_name))

        ax.legend(
            [h[0] for h in legend_handles],
            [h[1] for h in legend_handles],
            loc="upper left",
            fontsize=8,
            framealpha=0.9,
            title="Structures",
            title_fontsize=9,
        )

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

        # Try to load estimation data
        est_file = result_dir / f"est_{prompt_variant}.json"
        trajectory_scores = {}
        structures = []

        if est_file.exists():
            with open(est_file) as f:
                est_data = json.load(f)

            # Get structures and scores
            if est_data.get("systems"):
                system = est_data["systems"][0]  # Use first system
                structures = [s["structure"] for s in system["structures"]]

                # Map trajectory text to scores
                for i, traj in enumerate(trajectories):
                    scores = [s["scores"][i] for s in system["structures"]]
                    trajectory_scores[traj["text"]] = scores

            print(f"  Loaded {len(structures)} structures from estimation data")
        else:
            print("  No estimation data found, skipping structure coloring")

        # Build and plot word tree
        word_tree = build_word_tree(trajectories, trajectory_scores)
        plot_tree(
            word_tree,
            "Word Tree",
            output_dir / "word_tree.png",
            structures=structures,
            max_depth=10,
            min_count=1,
        )

        # Build and plot phrase tree
        phrase_tree = build_phrase_tree(trajectories, trajectory_scores, chunk_size=3)
        plot_tree(
            phrase_tree,
            "Phrase Tree",
            output_dir / "phrase_tree.png",
            structures=structures,
            max_depth=6,
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

    viz_dir = result_dir / "viz"

    print(f"Visualizing: {result_dir}")
    print(f"Output: {viz_dir}")

    visualize_results(result_dir, viz_dir)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
