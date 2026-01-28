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

    fig, ax = plt.subplots(figsize=(max(12, (depth + 1) * 3), max(6, n_leaves * 0.8)))

    _draw_edges(ax, edges)
    _draw_nodes(ax, nodes, scores, len(structures))

    if structures:
        _draw_legend(ax, structures)

    # Title with styled greedy text below
    if greedy_parts:
        ax.set_title(title, fontsize=14, fontweight="bold")
        _draw_styled_text(fig, greedy_parts)
    else:
        ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    x_range = max(all_x) - min(all_x) if len(set(all_x)) > 1 else 1
    y_range = max(all_y) - min(all_y) if len(set(all_y)) > 1 else 1
    ax.set_xlim(min(all_x) - 0.5, max(all_x) + x_range * 0.15 + 3)
    ax.set_ylim(
        min(all_y) - max(y_range * 0.1, 1.0), max(all_y) + max(y_range * 0.1, 0.5)
    )

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


def _draw_styled_text(
    fig, parts: list[tuple[str, str]], max_chars_per_line: int = 100
) -> None:
    """Draw styled text below the title using multiple text elements, with word wrapping."""
    # Style definitions for each part type
    styles = {
        TEXT_PART_TEMPLATE: {"color": "#999999", "fontsize": 6, "fontweight": "normal"},
        TEXT_PART_PROMPT: {"color": "#222222", "fontsize": 7, "fontweight": "bold"},
        TEXT_PART_CONTINUATION: {
            "color": "#666666",
            "fontsize": 7,
            "fontweight": "normal",
        },
    }

    # Escape newlines for display
    parts = [(text.replace("\n", "↵"), part_type) for text, part_type in parts]

    # Build lines with word wrapping
    lines = []  # Each line is a list of (text, part_type) tuples
    current_line = []
    current_len = 0

    for text, part_type in parts:
        # Split text into chunks that fit
        remaining = text
        while remaining:
            space_left = max_chars_per_line - current_len
            if len(remaining) <= space_left:
                current_line.append((remaining, part_type))
                current_len += len(remaining)
                remaining = ""
            else:
                # Add what fits to current line
                if space_left > 0:
                    current_line.append((remaining[:space_left], part_type))
                # Start new line with rest
                lines.append(current_line)
                current_line = []
                current_len = 0
                remaining = remaining[space_left:]

    if current_line:
        lines.append(current_line)

    # Limit to 3 lines max
    if len(lines) > 3:
        lines = lines[:3]
        # Add ellipsis to last line
        if lines[-1]:
            last_text, last_type = lines[-1][-1]
            lines[-1][-1] = (last_text + "...", last_type)

    # Draw each line
    y_pos = 0.945
    line_height = 0.018

    for line_parts in lines:
        # Calculate line width for centering
        line_len = sum(len(text) for text, _ in line_parts)
        x_pos = 0.5 - (line_len * 0.0028)  # Rough centering
        x_pos = max(0.02, x_pos)

        for text, part_type in line_parts:
            style = styles.get(part_type, styles[TEXT_PART_PROMPT])
            fig.text(
                x_pos,
                y_pos,
                text,
                fontfamily="monospace",
                ha="left",
                va="top",
                transform=fig.transFigure,
                **style,
            )
            # Advance position (rough character width estimate based on fontsize)
            char_width = 0.0056 if style["fontsize"] >= 7 else 0.0048
            x_pos += len(text) * char_width

        y_pos -= line_height


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
    """Draw structure legend."""
    handles = [
        (
            plt.Rectangle((0, 0), 1, 1, fc=COLORS[i % len(COLORS)], ec="black", lw=0.5),
            s[:30] + "..." if len(s) > 30 else s,
        )
        for i, s in enumerate(structures)
    ]
    ax.legend(
        [h[0] for h in handles],
        [h[1] for h in handles],
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
        title="Structures",
    )


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
