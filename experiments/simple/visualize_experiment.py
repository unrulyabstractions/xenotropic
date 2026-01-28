#!/usr/bin/env python3
"""
Visualize experiment results as trajectory trees.

Creates visualizations of trajectories as:
1. Word tree - each node is a word
2. Phrase tree - each node is a phrase chunk
3. Token tree - each node is a BPE token (when available)

Edge labels show true conditional probability P(token|context).
Edge thickness is proportional to relative probability at branch point.
Leaf nodes are colored by dominant structure compliance.

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
    probability: float  # Conditional probability P(token|context) if available
    children: dict[str, TreeNode] = field(default_factory=dict)
    count: int = 1  # Number of trajectories through this node
    has_true_prob: bool = False  # True if probability is actual conditional prob
    # Layout fields (set during layout computation)
    x: float = 0.0
    y: float = 0.0
    # Structure compliance (only for leaf nodes)
    structure_scores: Optional[list[float]] = None
    trajectory_text: Optional[str] = None  # Full trajectory text for leaf nodes
    is_greedy: bool = False  # True if this is the greedy trajectory leaf
    # Trajectories passing through this node (for computing core at branching points)
    trajectory_probs: list[tuple[str, float]] = field(
        default_factory=list
    )  # [(text, prob), ...]

    @classmethod
    def create_root(cls, label: str = "<root>") -> TreeNode:
        return cls(label=label, probability=1.0, has_true_prob=True)

    def is_leaf(self) -> bool:
        return len(self.children) == 0


# -----------------------------------------------------------------------------
# Tree Building
# -----------------------------------------------------------------------------


def has_token_data(trajectories: list[dict]) -> bool:
    """Check if trajectories have actual token-level data."""
    for traj in trajectories:
        if traj.get("per_token_logprobs"):
            return True
    return False


def build_token_tree(
    trajectories: list[dict],
    trajectory_scores: Optional[dict[str, list[float]]] = None,
    prompt: str = "<root>",
) -> Optional[TreeNode]:
    """Build a tree from token sequences. Returns None if no token data."""
    if not has_token_data(trajectories):
        return None

    root = TreeNode.create_root(label=prompt)

    for traj in trajectories:
        traj_prob = traj["probability"]
        text = traj["text"]
        tokens = traj.get("per_token_logprobs", [])
        is_greedy = traj.get("is_greedy", False)

        if not tokens:
            continue

        current = root
        # Track trajectory through root
        root.trajectory_probs.append((text, traj_prob))

        for i, tok_info in enumerate(tokens):
            token = tok_info["token"]
            logprob = tok_info.get("logprob", 0)
            cond_prob = np.exp(logprob) if logprob else traj_prob ** (1 / len(tokens))
            is_last = i == len(tokens) - 1

            if token not in current.children:
                current.children[token] = TreeNode(
                    label=token,
                    probability=cond_prob,
                    count=0,
                    has_true_prob=True,
                )

            current.children[token].count += 1
            current.children[token].probability = cond_prob
            # Track trajectory through this node
            current.children[token].trajectory_probs.append((text, traj_prob))

            # If this is a leaf node, store structure scores and greedy flag
            if is_last:
                if trajectory_scores and text in trajectory_scores:
                    current.children[token].structure_scores = trajectory_scores[text]
                current.children[token].trajectory_text = text
                current.children[token].is_greedy = is_greedy

            current = current.children[token]

    return root


def _compute_word_probs(tokens: list[dict], text: str) -> list[tuple[str, float]]:
    """
    Compute conditional probability for each word by chaining token probabilities.

    P(word | context) = product of P(token_i | context, token_1..token_{i-1})
    for all tokens that make up the word.

    Returns list of (word, cond_prob) tuples.
    """
    if not tokens:
        return []

    # Reconstruct text from tokens to align with words
    token_texts = [t["token"] for t in tokens]
    token_logprobs = [t.get("logprob", 0) for t in tokens]

    # Split text into words
    words = re.findall(r"\S+", text)
    if not words:
        return []

    result = []
    token_idx = 0
    char_pos = 0

    for word in words:
        # Find where this word starts in the text
        word_start = text.find(word, char_pos)
        if word_start == -1:
            # Fallback: use uniform distribution
            result.append((word, 1.0 / len(words)))
            continue

        word_end = word_start + len(word)

        # Accumulate tokens that belong to this word
        word_log_prob = 0.0
        reconstructed = ""
        tokens_used = 0

        while token_idx < len(token_texts):
            tok = token_texts[token_idx]
            tok_stripped = tok.lstrip()  # Tokens often have leading space

            # Check if this token contributes to the current word
            if tok_stripped and word.startswith(reconstructed + tok_stripped):
                reconstructed += tok_stripped
                word_log_prob += token_logprobs[token_idx]
                token_idx += 1
                tokens_used += 1

                if reconstructed == word:
                    break
            elif tok.strip() == "":
                # Whitespace token, skip
                token_idx += 1
            else:
                # Token doesn't match - might be subword that spans words
                # Try including the token anyway if it starts the remaining part
                remaining = word[len(reconstructed) :]
                if remaining and tok_stripped.startswith(remaining):
                    word_log_prob += token_logprobs[token_idx]
                    token_idx += 1
                    break
                elif tok_stripped and remaining.startswith(tok_stripped):
                    reconstructed += tok_stripped
                    word_log_prob += token_logprobs[token_idx]
                    token_idx += 1
                    if reconstructed == word:
                        break
                else:
                    break

        # Convert log prob to probability
        cond_prob = np.exp(word_log_prob) if tokens_used > 0 else 1.0 / len(words)
        result.append((word, cond_prob))
        char_pos = word_end

    return result


def build_word_tree(
    trajectories: list[dict],
    trajectory_scores: Optional[dict[str, list[float]]] = None,
    prompt: str = "<root>",
) -> TreeNode:
    """Build a tree from word sequences with chained token probabilities."""
    root = TreeNode.create_root(label=prompt)

    for traj in trajectories:
        text = traj["text"]
        traj_prob = traj["probability"]
        is_greedy = traj.get("is_greedy", False)
        tokens = traj.get("per_token_logprobs", [])

        # Compute word conditional probabilities from token probs
        word_probs = _compute_word_probs(tokens, text)
        has_token_data = len(tokens) > 0

        if not word_probs:
            # Fallback to simple word splitting
            words = re.findall(r"\S+", text)
            prob = traj["probability"]
            word_prob = prob ** (1 / len(words)) if words else 1.0
            word_probs = [(w, word_prob) for w in words]

        current = root
        # Track trajectory through root
        root.trajectory_probs.append((text, traj_prob))

        for i, (word, word_cond_prob) in enumerate(word_probs):
            is_last = i == len(word_probs) - 1

            if word not in current.children:
                current.children[word] = TreeNode(
                    label=word,
                    probability=word_cond_prob,
                    count=0,
                    has_true_prob=has_token_data,
                )

            current.children[word].count += 1
            current.children[word].probability = word_cond_prob
            # Track trajectory through this node
            current.children[word].trajectory_probs.append((text, traj_prob))

            # If this is a leaf node, store structure scores and greedy flag
            if is_last:
                if trajectory_scores and text in trajectory_scores:
                    current.children[word].structure_scores = trajectory_scores[text]
                current.children[word].trajectory_text = text
                current.children[word].is_greedy = is_greedy

            current = current.children[word]

    return root


def build_phrase_tree(
    trajectories: list[dict],
    trajectory_scores: Optional[dict[str, list[float]]] = None,
    prompt: str = "<root>",
) -> TreeNode:
    """Build a tree that only branches at actual divergence points.

    Collapses linear chains of words into single phrase nodes.
    """
    # First build word tree
    word_tree = build_word_tree(trajectories, trajectory_scores, prompt)

    # Then collapse linear chains
    return _collapse_linear_chains(word_tree)


def _collapse_linear_chains(node: TreeNode) -> TreeNode:
    """Collapse linear chains (single-child nodes) into phrase nodes.

    The probability of the collapsed node is the product of all probabilities
    in the chain: P(phrase | parent) = P(w1|parent) * P(w2|w1) * ... * P(wn|w_{n-1})
    """
    # Collect labels and probabilities along linear chain
    labels = [node.label]
    probs = [node.probability]
    current = node
    has_true_prob = node.has_true_prob

    # Follow single-child chain
    while len(current.children) == 1:
        child = list(current.children.values())[0]
        labels.append(child.label)
        probs.append(child.probability)
        has_true_prob = has_true_prob and child.has_true_prob
        current = child

    # Combined probability is product of chain probabilities
    combined_prob = np.prod(probs)

    # Create collapsed node with combined label
    combined_label = " ".join(labels)
    collapsed = TreeNode(
        label=combined_label,
        probability=combined_prob,
        count=current.count,
        has_true_prob=has_true_prob,
        structure_scores=current.structure_scores,
        trajectory_text=current.trajectory_text,
        is_greedy=current.is_greedy,
        trajectory_probs=current.trajectory_probs,  # Use final node's trajectory_probs
    )

    # Recursively collapse children
    for child in current.children.values():
        collapsed_child = _collapse_linear_chains(child)
        collapsed.children[collapsed_child.label] = collapsed_child

    return collapsed


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

    # Sort children by probability (highest first, at top - lower y values are higher on screen)
    sorted_children = sorted(node.children.values(), key=lambda c: c.probability)

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
            has_true_prob=node.has_true_prob,
            structure_scores=node.structure_scores,
            trajectory_text=node.trajectory_text,
            is_greedy=node.is_greedy,
            trajectory_probs=node.trajectory_probs,
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
        has_true_prob=node.has_true_prob,
        structure_scores=node.structure_scores,
        trajectory_text=node.trajectory_text,
        is_greedy=node.is_greedy,
        trajectory_probs=node.trajectory_probs,
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


def _compute_node_core(
    node: TreeNode,
    trajectory_scores: dict[str, list[float]],
    num_structures: int,
) -> Optional[list[float]]:
    """
    Compute core values for each structure at a branching node.

    Core = sum(prob_i * score_i) / sum(prob_i) for trajectories through this node.
    Returns list of core values per structure, or None if no data.
    """
    if not node.trajectory_probs or not trajectory_scores:
        return None

    # Gather scores and probabilities for trajectories through this node
    total_prob = 0.0
    weighted_scores = [0.0] * num_structures

    for text, prob in node.trajectory_probs:
        if text in trajectory_scores:
            scores = trajectory_scores[text]
            total_prob += prob
            for i, score in enumerate(scores):
                weighted_scores[i] += prob * score

    if total_prob == 0:
        return None

    # Normalize to get core
    return [ws / total_prob for ws in weighted_scores]


def plot_tree(
    root: TreeNode,
    title: str,
    output_path: Path,
    structures: list[str],
    trajectory_scores: Optional[dict[str, list[float]]] = None,
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

        # Add true conditional probability label (only if we have real logprob data)
        if child_node.has_true_prob:
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2

            cond_prob = child_node.probability
            # Use scientific notation for small probabilities
            if cond_prob < 0.01:
                prob_text = f"{cond_prob:.1e}"
            else:
                prob_text = f"{cond_prob:.2f}"
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
        is_branching = len(node.children) > 1

        if is_leaf and node.structure_scores is not None:
            # Color leaf by dominant structure
            dominant_idx = int(np.argmax(node.structure_scores))
            node_color = PASTEL_COLORS[dominant_idx % len(PASTEL_COLORS)]
        elif is_branching and trajectory_scores and len(structures) > 0:
            # Color branching node by dominant structure in core
            core = _compute_node_core(node, trajectory_scores, len(structures))
            if core is not None:
                dominant_idx = int(np.argmax(core))
                node_color = PASTEL_COLORS[dominant_idx % len(PASTEL_COLORS)]
            else:
                node_color = "#DDDDDD"
        else:
            node_color = "#DDDDDD"  # Gray for other nodes

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

        # Node label (below node)
        max_label_len = 18
        display_label = (
            node.label[:max_label_len] + "..."
            if len(node.label) > max_label_len
            else node.label
        )
        display_label = display_label.replace("\n", "\\n")

        # Label below node
        ax.annotate(
            display_label,
            (pos[0], pos[1] - 0.3),
            fontsize=8,
            ha="center",
            va="top",
            fontfamily="monospace",
            zorder=4,
        )

        # Add "greedy" indicator to the right for greedy leaf nodes
        if is_leaf and node.is_greedy:
            ax.annotate(
                "greedy",
                (pos[0] + 0.5, pos[1]),
                fontsize=7,
                ha="left",
                va="center",
                fontfamily="monospace",
                fontweight="bold",
                color="#CD7F32",  # Red-gold / bronze color
                zorder=4,
            )

        # For leaf nodes with structure scores, show colored compliance values below label
        if is_leaf and node.structure_scores is not None:
            # Build score string with colored parts
            score_y = pos[1] - 0.55

            # Draw bracket and scores
            score_parts = []
            for i, score in enumerate(node.structure_scores):
                score_parts.append(f"{score:.2f}")

            full_text = "[" + ", ".join(score_parts) + "]"

            # Draw each score with its color
            # First calculate total width for centering
            ax.annotate(
                "[",
                (pos[0] - len(full_text) * 0.03, score_y),
                fontsize=7,
                ha="left",
                va="top",
                fontfamily="monospace",
                zorder=4,
            )

            current_x = pos[0] - len(full_text) * 0.03 + 0.06
            for i, score in enumerate(node.structure_scores):
                color = PASTEL_COLORS[i % len(PASTEL_COLORS)]
                # Darken for text readability
                darker = tuple(int(c * 0.6) for c in bytes.fromhex(color[1:]))
                text_color = f"#{darker[0]:02x}{darker[1]:02x}{darker[2]:02x}"

                score_str = f"{score:.2f}"
                ax.annotate(
                    score_str,
                    (current_x, score_y),
                    fontsize=7,
                    ha="left",
                    va="top",
                    fontfamily="monospace",
                    color=text_color,
                    fontweight="bold",
                    zorder=4,
                )
                current_x += len(score_str) * 0.06

                if i < len(node.structure_scores) - 1:
                    ax.annotate(
                        ", ",
                        (current_x, score_y),
                        fontsize=7,
                        ha="left",
                        va="top",
                        fontfamily="monospace",
                        zorder=4,
                    )
                    current_x += 0.12

            ax.annotate(
                "]",
                (current_x, score_y),
                fontsize=7,
                ha="left",
                va="top",
                fontfamily="monospace",
                zorder=4,
            )

    # Title
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")

    # Add margins
    x_margin = x_range * 0.15 + 3
    y_margin_top = max(y_range * 0.1, 0.5)
    y_margin_bottom = max(
        y_range * 0.1, 1.0
    )  # Extra margin at bottom for structure scores
    ax.set_xlim(min(all_x) - 0.5, max(all_x) + x_margin)
    ax.set_ylim(min(all_y) - y_margin_bottom, max(all_y) + y_margin_top)

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
        prompt_text = gen_data.get("prompt_text", "<root>")

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
        word_tree = build_word_tree(trajectories, trajectory_scores, prompt_text)
        plot_tree(
            word_tree,
            "Word Tree",
            output_dir / "word_tree.png",
            structures=structures,
            trajectory_scores=trajectory_scores,
            max_depth=10,
            min_count=1,
        )

        # Build and plot phrase tree
        phrase_tree = build_phrase_tree(trajectories, trajectory_scores, prompt_text)
        plot_tree(
            phrase_tree,
            "Phrase Tree",
            output_dir / "phrase_tree.png",
            structures=structures,
            trajectory_scores=trajectory_scores,
            max_depth=6,
            min_count=1,
        )

        # Build and plot token tree (only if we have token data)
        token_tree = build_token_tree(trajectories, trajectory_scores, prompt_text)
        if token_tree is not None:
            plot_tree(
                token_tree,
                "Token Tree",
                output_dir / "token_tree.png",
                structures=structures,
                trajectory_scores=trajectory_scores,
                max_depth=15,
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
