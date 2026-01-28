"""Tree plotting and layout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from trees import TreeNode, build_tree

COLORS = ["#FFB3BA", "#BAFFC9", "#BAE1FF", "#FFFFBA", "#FFDFba", "#E0BBE4"]

# Text part types for styled rendering
TEXT_PART_TEMPLATE = "template"  # Chat template prefix/suffix - gray, smaller
TEXT_PART_PROMPT = "prompt"  # User's prompt - normal
TEXT_PART_CONTINUATION = "continuation"  # Model continuation - gray


def visualize_experiment(result_dir: Path, output_dir: Path | None = None) -> None:
    """Visualize a single experiment's results."""
    if output_dir is None:
        output_dir = result_dir / "viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    for gen_file in result_dir.glob("gen_*.json"):
        print(f"  {gen_file.name}")

        with open(gen_file) as f:
            data = json.load(f)

        trajectories = data["trajectories"]
        prompt = data.get("prompt_text", "<root>")
        formatted_prompt = data.get("formatted_prompt", prompt)
        variant = data["prompt_variant"]

        if not trajectories:
            print("    No trajectories")
            continue

        scores, structures = _load_scores(result_dir, variant, trajectories)
        greedy_traj = next((t for t in trajectories if t.get("is_greedy")), None)
        # Build text parts for styled rendering
        greedy_parts = None
        if greedy_traj:
            continuation = greedy_traj["text"]
            greedy_parts = _build_text_parts(prompt, formatted_prompt, continuation)

        for mode in ["word", "phrase", "token"]:
            tree = build_tree(trajectories, scores, prompt, mode)
            if tree:
                plot_tree(
                    tree,
                    f"{mode.title()} Tree",
                    output_dir / f"{mode}_tree.png",
                    structures,
                    scores,
                    greedy_parts=greedy_parts,
                )
            elif mode == "token":
                print("    Skipping token tree (no token data)")


def _load_scores(
    result_dir: Path, variant: str, trajectories: list[dict]
) -> tuple[dict[str, list[float]], list[str]]:
    """Load structure scores from estimation file."""
    scores, structures = {}, []
    est_file = result_dir / f"est_{variant}.json"

    if est_file.exists():
        with open(est_file) as f:
            est = json.load(f)
        if est.get("systems"):
            sys_data = est["systems"][0]
            structures = [s["structure"] for s in sys_data["structures"]]
            for i, t in enumerate(trajectories):
                scores[t["text"]] = [s["scores"][i] for s in sys_data["structures"]]
        print(f"    {len(structures)} structures")

    return scores, structures


def _build_text_parts(
    prompt: str, formatted_prompt: str, continuation: str
) -> list[tuple[str, str]]:
    """
    Build list of (text, part_type) tuples for styled rendering.

    Splits formatted_prompt into template prefix, user prompt, template suffix,
    then adds continuation.
    """
    parts = []

    # Find where the user prompt appears in the formatted prompt
    prompt_start = formatted_prompt.find(prompt)

    if prompt_start == -1:
        # Prompt not found in formatted - just return the whole thing as template
        parts.append((formatted_prompt, TEXT_PART_TEMPLATE))
        parts.append((continuation, TEXT_PART_CONTINUATION))
        return parts

    # Template prefix (before user prompt)
    if prompt_start > 0:
        prefix = formatted_prompt[:prompt_start]
        parts.append((prefix, TEXT_PART_TEMPLATE))

    # User prompt
    parts.append((prompt, TEXT_PART_PROMPT))

    # Template suffix (after user prompt)
    prompt_end = prompt_start + len(prompt)
    if prompt_end < len(formatted_prompt):
        suffix = formatted_prompt[prompt_end:]
        parts.append((suffix, TEXT_PART_TEMPLATE))

    # Model continuation
    parts.append((continuation, TEXT_PART_CONTINUATION))

    return parts


def plot_tree(
    root: TreeNode,
    title: str,
    path: Path,
    structures: list[str],
    scores: dict[str, list[float]],
    greedy_parts: list[tuple[str, str]] | None = None,
) -> None:
    """Render tree to PNG."""
    _layout(root)
    edges, nodes = _collect(root)

    if not nodes:
        return

    all_x = [pos[0] for _, pos in nodes]
    all_y = [pos[1] for _, pos in nodes]
    n_leaves = sum(1 for n, _ in nodes if n.is_leaf())
    depth = max(all_x) / 2.5 if all_x else 1

    # Calculate figure dimensions
    tree_width = max(10, (depth + 1) * 2.5)
    tree_height = max(4, n_leaves * 0.7)

    # Add space for header (title + styled text) and legend
    header_height = 1.2 if greedy_parts else 0.6
    legend_height = 0.8 if structures else 0
    total_height = tree_height + header_height + legend_height

    fig = plt.figure(figsize=(tree_width, total_height))

    # Create grid: header at top, tree in middle, legend at bottom
    if structures:
        gs = fig.add_gridspec(
            3,
            1,
            height_ratios=[header_height, tree_height, legend_height],
            hspace=0.05,
        )
        ax_header = fig.add_subplot(gs[0])
        ax_tree = fig.add_subplot(gs[1])
        ax_legend = fig.add_subplot(gs[2])
    else:
        gs = fig.add_gridspec(
            2, 1, height_ratios=[header_height, tree_height], hspace=0.05
        )
        ax_header = fig.add_subplot(gs[0])
        ax_tree = fig.add_subplot(gs[1])
        ax_legend = None

    # Draw header (title + styled text)
    ax_header.axis("off")
    ax_header.text(
        0.5,
        0.7,
        title,
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="center",
        transform=ax_header.transAxes,
    )
    if greedy_parts:
        _draw_styled_text(ax_header, greedy_parts, y_start=0.4)

    # Draw tree
    ax_tree.axis("off")
    _draw_edges(ax_tree, edges)
    _draw_nodes(ax_tree, nodes, scores, len(structures))

    x_range = max(all_x) - min(all_x) if len(set(all_x)) > 1 else 1
    y_range = max(all_y) - min(all_y) if len(set(all_y)) > 1 else 1
    ax_tree.set_xlim(min(all_x) - 0.5, max(all_x) + x_range * 0.12 + 2)
    ax_tree.set_ylim(min(all_y) - 0.8, max(all_y) + 0.5)

    # Draw legend in dedicated area
    if structures and ax_legend:
        _draw_legend(ax_legend, structures)

    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


def _draw_styled_text(
    ax,
    parts: list[tuple[str, str]],
    y_start: float = 0.5,
    max_total_chars: int = 200,
) -> None:
    """Draw styled text centered in axes, with truncation if needed."""
    # Colors for each part type
    colors = {
        TEXT_PART_TEMPLATE: "#999999",
        TEXT_PART_PROMPT: "#000000",
        TEXT_PART_CONTINUATION: "#666666",
    }

    # Escape newlines
    parts = [(text.replace("\n", "↵"), part_type) for text, part_type in parts]

    # Truncate if total length exceeds max
    total_len = sum(len(text) for text, _ in parts)
    if total_len > max_total_chars:
        remaining = max_total_chars - 3
        new_parts = []
        for text, ptype in parts:
            if remaining <= 0:
                break
            if len(text) <= remaining:
                new_parts.append((text, ptype))
                remaining -= len(text)
            else:
                new_parts.append((text[:remaining] + "...", ptype))
                break
        parts = new_parts

    # Draw each part with its color, all on one or two lines
    full_text = "".join(text for text, _ in parts)

    # Split into 2 lines if too long
    if len(full_text) > 100:
        mid = len(full_text) // 2
        # Try to split at a space or special char
        for i in range(mid, min(mid + 20, len(full_text))):
            if full_text[i] in " ↵<>":
                mid = i
                break
        line1, line2 = full_text[:mid], full_text[mid:]

        ax.text(
            0.5,
            y_start,
            line1,
            fontfamily="monospace",
            fontsize=7,
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#666666",
        )
        ax.text(
            0.5,
            y_start - 0.35,
            line2,
            fontfamily="monospace",
            fontsize=7,
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#666666",
        )
    else:
        ax.text(
            0.5,
            y_start,
            full_text,
            fontfamily="monospace",
            fontsize=7,
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#666666",
        )


def _layout(
    node: TreeNode, depth: int = 0, y: float = 0, x_sp: float = 2.5, y_sp: float = 1.2
) -> float:
    """Compute x,y positions. Returns height used."""
    node.x = depth * x_sp

    if node.is_leaf():
        node.y = y
        return 1.0

    children = sorted(node.children.values(), key=lambda c: c.prob)
    curr_y, positions = y, []

    for child in children:
        h = _layout(child, depth + 1, curr_y, x_sp, y_sp)
        positions.append(child.y)
        curr_y += h * y_sp

    node.y = (positions[0] + positions[-1]) / 2 if positions else y
    return max(len(positions), 1)


def _collect(node: TreeNode) -> tuple[list, list]:
    """Collect edges and nodes for plotting."""
    edges, nodes = [], [(node, (node.x, node.y))]

    if node.children:
        total = sum(c.prob for c in node.children.values())
        for child in node.children.values():
            edges.append(((node.x, node.y), (child.x, child.y), child, total))
            ce, cn = _collect(child)
            edges.extend(ce)
            nodes.extend(cn)

    return edges, nodes


def _draw_edges(ax, edges: list) -> None:
    """Draw tree edges with probability labels."""
    for (x1, y1), (x2, y2), child, total in edges:
        rel = child.prob / total if total else 0.5
        ax.plot(
            [x1, x2],
            [y1, y2],
            color="#666",
            linewidth=0.5 + rel * 4,
            alpha=0.6,
            zorder=1,
        )

        if child.has_true_prob:
            ax.annotate(
                _fmt_prob(child.prob),
                ((x1 + x2) / 2, (y1 + y2) / 2),
                fontsize=7,
                ha="center",
                va="center",
                color="#333",
                bbox=dict(
                    boxstyle="round,pad=0.1",
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none",
                ),
                zorder=3,
            )


def _draw_nodes(ax, nodes: list, scores: dict, n_structs: int) -> None:
    """Draw tree nodes with labels and scores."""
    for node, (x, y) in nodes:
        color = _node_color(node, scores, n_structs)
        ax.scatter(
            [x],
            [y],
            s=60 + node.count * 30,
            c=[color],
            alpha=0.9,
            edgecolors="black",
            linewidths=0.8,
            zorder=2,
        )

        label = node.label[:18] + "..." if len(node.label) > 18 else node.label
        ax.annotate(
            label.replace("\n", "\\n"),
            (x, y - 0.3),
            fontsize=8,
            ha="center",
            va="top",
            fontfamily="monospace",
            zorder=4,
        )

        if node.is_leaf() and node.is_greedy:
            ax.annotate(
                "greedy",
                (x + 0.5, y),
                fontsize=7,
                ha="left",
                va="center",
                fontfamily="monospace",
                fontweight="bold",
                color="#CD7F32",
                zorder=4,
            )

        if node.is_leaf() and node.scores:
            _draw_scores(ax, x, y, node.scores)


def _draw_scores(ax, x: float, y: float, node_scores: list[float]) -> None:
    """Draw score array below node."""
    score_x = x - len(node_scores) * 0.15
    ax.annotate(
        "[",
        (score_x, y - 0.55),
        fontsize=7,
        ha="left",
        va="top",
        fontfamily="monospace",
    )
    score_x += 0.06

    for i, s in enumerate(node_scores):
        c = _darken(COLORS[i % len(COLORS)])
        ax.annotate(
            f"{s:.2f}",
            (score_x, y - 0.55),
            fontsize=7,
            ha="left",
            va="top",
            fontfamily="monospace",
            color=c,
            fontweight="bold",
        )
        score_x += 0.3
        if i < len(node_scores) - 1:
            ax.annotate(
                ", ",
                (score_x, y - 0.55),
                fontsize=7,
                ha="left",
                va="top",
                fontfamily="monospace",
            )
            score_x += 0.1

    ax.annotate(
        "]",
        (score_x, y - 0.55),
        fontsize=7,
        ha="left",
        va="top",
        fontfamily="monospace",
    )


def _draw_legend(ax, structures: list[str]) -> None:
    """Draw structure legend horizontally centered in its own axes area."""
    ax.axis("off")

    n = len(structures)
    if n == 0:
        return

    # Estimate total width needed (box + label for each structure)
    labels = [s[:30] + "..." if len(s) > 30 else s for s in structures]
    # Approximate width: box (0.015) + gap (0.01) + chars * 0.006 + padding (0.03)
    item_widths = [0.015 + 0.01 + len(label) * 0.006 + 0.03 for label in labels]
    total_width = sum(item_widths)

    # Start from center-left
    start_x = 0.5 - total_width / 2
    x = start_x

    for i, (s, label) in enumerate(zip(structures, labels)):
        color = COLORS[i % len(COLORS)]

        # Draw colored box
        ax.add_patch(
            plt.Rectangle(
                (x, 0.25),
                0.015,
                0.5,
                fc=color,
                ec="black",
                lw=0.5,
                transform=ax.transAxes,
                clip_on=False,
            )
        )

        # Draw label
        ax.text(
            x + 0.02,
            0.5,
            label,
            fontsize=7,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )

        x += item_widths[i]


def _node_color(node: TreeNode, scores: dict, n_structs: int) -> str:
    """Get node color based on dominant structure."""
    if node.is_leaf() and node.scores:
        return COLORS[int(np.argmax(node.scores)) % len(COLORS)]

    if node.is_branching() and scores and n_structs:
        core = _compute_core(node, scores, n_structs)
        if core:
            return COLORS[int(np.argmax(core)) % len(COLORS)]

    return "#DDDDDD"


def _compute_core(node: TreeNode, scores: dict, n: int) -> Optional[list[float]]:
    """Compute probability-weighted core at a node."""
    if not node.traj_probs:
        return None

    total, weighted = 0.0, [0.0] * n
    for text, prob in node.traj_probs:
        if text in scores:
            total += prob
            for i, s in enumerate(scores[text]):
                weighted[i] += prob * s

    return [w / total for w in weighted] if total else None


def _fmt_prob(p: float) -> str:
    return f"{p:.1e}" if p < 0.01 else f"{p:.2f}"


def _darken(hex_color: str) -> str:
    rgb = bytes.fromhex(hex_color[1:])
    return f"#{int(rgb[0] * 0.6):02x}{int(rgb[1] * 0.6):02x}{int(rgb[2] * 0.6):02x}"
