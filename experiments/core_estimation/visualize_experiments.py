"""Visualize experiment results as trajectory trees."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

PASTEL_COLORS = ["#FFB3BA", "#BAFFC9", "#BAE1FF", "#FFFFBA", "#FFDFba", "#E0BBE4"]


# -----------------------------------------------------------------------------
# Tree Node
# -----------------------------------------------------------------------------


@dataclass
class TreeNode:
    """Node in a trajectory tree."""

    label: str
    prob: float = 1.0
    children: dict[str, TreeNode] = field(default_factory=dict)
    count: int = 1
    has_true_prob: bool = False
    scores: Optional[list[float]] = None
    is_greedy: bool = False
    traj_probs: list[tuple[str, float]] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0

    def is_leaf(self) -> bool:
        return not self.children

    def is_branching(self) -> bool:
        return len(self.children) > 1


# -----------------------------------------------------------------------------
# Tree Building
# -----------------------------------------------------------------------------


def build_token_tree(
    trajectories: list[dict],
    scores: dict[str, list[float]],
    prompt: str,
) -> Optional[TreeNode]:
    """Build tree from BPE tokens. Returns None if no token data."""
    if not any(t.get("per_token_logprobs") for t in trajectories):
        return None

    root = TreeNode(label=prompt, has_true_prob=True)

    for traj in trajectories:
        tokens = traj.get("per_token_logprobs", [])
        if not tokens:
            continue

        text, prob = traj["text"], traj["probability"]
        root.traj_probs.append((text, prob))
        current = root

        for i, tok in enumerate(tokens):
            token, logprob = tok["token"], tok.get("logprob", 0)
            cond_prob = np.exp(logprob) if logprob else prob ** (1 / len(tokens))

            if token not in current.children:
                current.children[token] = TreeNode(
                    label=token, prob=cond_prob, count=0, has_true_prob=True
                )

            child = current.children[token]
            child.count += 1
            child.prob = cond_prob
            child.traj_probs.append((text, prob))

            if i == len(tokens) - 1:  # Leaf
                child.scores = scores.get(text)
                child.is_greedy = traj.get("is_greedy", False)

            current = child

    return root


def build_word_tree(
    trajectories: list[dict],
    scores: dict[str, list[float]],
    prompt: str,
) -> TreeNode:
    """Build tree from whitespace-split words."""
    root = TreeNode(label=prompt, has_true_prob=True)

    for traj in trajectories:
        text, prob = traj["text"], traj["probability"]
        tokens = traj.get("per_token_logprobs", [])
        word_probs = _compute_word_probs(tokens, text)
        has_tokens = bool(tokens)

        root.traj_probs.append((text, prob))
        current = root

        for i, (word, word_prob) in enumerate(word_probs):
            if word not in current.children:
                current.children[word] = TreeNode(
                    label=word, prob=word_prob, count=0, has_true_prob=has_tokens
                )

            child = current.children[word]
            child.count += 1
            child.prob = word_prob
            child.traj_probs.append((text, prob))

            if i == len(word_probs) - 1:  # Leaf
                child.scores = scores.get(text)
                child.is_greedy = traj.get("is_greedy", False)

            current = child

    return root


def build_phrase_tree(
    trajectories: list[dict],
    scores: dict[str, list[float]],
    prompt: str,
) -> TreeNode:
    """Build tree collapsed at branching points only."""
    return _collapse_chains(build_word_tree(trajectories, scores, prompt))


def _compute_word_probs(tokens: list[dict], text: str) -> list[tuple[str, float]]:
    """Compute P(word|context) by chaining token probabilities."""
    words = re.findall(r"\S+", text)
    if not words:
        return []
    if not tokens:
        p = 1.0 / len(words)
        return [(w, p) for w in words]

    token_texts = [t["token"] for t in tokens]
    token_lps = [t.get("logprob", 0) for t in tokens]

    result, token_idx, char_pos = [], 0, 0

    for word in words:
        word_start = text.find(word, char_pos)
        if word_start == -1:
            result.append((word, 1.0 / len(words)))
            continue

        log_prob, reconstructed = 0.0, ""

        while token_idx < len(token_texts) and reconstructed != word:
            tok = token_texts[token_idx].lstrip()
            if tok and word[len(reconstructed) :].startswith(tok):
                reconstructed += tok
                log_prob += token_lps[token_idx]
                token_idx += 1
            elif not token_texts[token_idx].strip():
                token_idx += 1
            else:
                break

        result.append((word, np.exp(log_prob) if log_prob else 1.0 / len(words)))
        char_pos = word_start + len(word)

    return result


def _collapse_chains(node: TreeNode) -> TreeNode:
    """Collapse single-child chains into phrase nodes."""
    labels, probs = [node.label], [node.prob]
    current, has_true = node, node.has_true_prob

    while len(current.children) == 1:
        child = next(iter(current.children.values()))
        labels.append(child.label)
        probs.append(child.prob)
        has_true = has_true and child.has_true_prob
        current = child

    collapsed = TreeNode(
        label=" ".join(labels),
        prob=float(np.prod(probs)),
        count=current.count,
        has_true_prob=has_true,
        scores=current.scores,
        is_greedy=current.is_greedy,
        traj_probs=current.traj_probs,
    )

    for child in current.children.values():
        c = _collapse_chains(child)
        collapsed.children[c.label] = c

    return collapsed


# -----------------------------------------------------------------------------
# Layout & Rendering
# -----------------------------------------------------------------------------


def _layout(
    node: TreeNode, depth: int = 0, y: float = 0, x_sp: float = 2.5, y_sp: float = 1.2
) -> float:
    """Compute x,y positions. Returns height used."""
    node.x = depth * x_sp

    if node.is_leaf():
        node.y = y
        return 1.0

    children = sorted(node.children.values(), key=lambda c: c.prob)  # High prob at top
    curr_y, positions = y, []

    for child in children:
        h = _layout(child, depth + 1, curr_y, x_sp, y_sp)
        positions.append(child.y)
        curr_y += h * y_sp

    node.y = (positions[0] + positions[-1]) / 2 if positions else y
    return max(sum(1 for _ in positions) or 1, 1.0)


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


def _node_color(node: TreeNode, scores: dict, n_structs: int) -> str:
    """Get node color based on dominant structure."""
    if node.is_leaf() and node.scores:
        return PASTEL_COLORS[int(np.argmax(node.scores)) % len(PASTEL_COLORS)]

    if node.is_branching() and scores and n_structs:
        core = _compute_core(node, scores, n_structs)
        if core:
            return PASTEL_COLORS[int(np.argmax(core)) % len(PASTEL_COLORS)]

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


def plot_tree(
    root: TreeNode,
    title: str,
    path: Path,
    structures: list[str],
    scores: dict[str, list[float]],
) -> None:
    """Render tree to PNG."""
    _layout(root)
    edges, nodes = _collect(root)

    if not nodes:
        return

    all_x = [n[1][0] for n in nodes]
    all_y = [n[1][1] for n in nodes]
    n_leaves = sum(1 for n, _ in nodes if n.is_leaf())
    depth = max(all_x) / 2.5 if all_x else 1

    fig, ax = plt.subplots(figsize=(max(12, (depth + 1) * 3), max(6, n_leaves * 0.8)))

    # Edges
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

    # Nodes
    for node, (x, y) in nodes:
        color = _node_color(node, scores, len(structures))
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

        # Label
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

        # Greedy marker
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

        # Structure scores
        if node.is_leaf() and node.scores:
            score_x = x - len(node.scores) * 0.15
            ax.annotate(
                "[",
                (score_x, y - 0.55),
                fontsize=7,
                ha="left",
                va="top",
                fontfamily="monospace",
            )
            score_x += 0.06
            for i, s in enumerate(node.scores):
                c = _darken(PASTEL_COLORS[i % len(PASTEL_COLORS)])
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
                if i < len(node.scores) - 1:
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

    # Legend
    if structures:
        handles = [
            (
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    fc=PASTEL_COLORS[i % len(PASTEL_COLORS)],
                    ec="black",
                    lw=0.5,
                ),
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


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------


def visualize_results(result_dir: Path, output_dir: Path) -> None:
    """Visualize all experiment results in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for gen_file in result_dir.glob("gen_*.json"):
        print(f"\nProcessing: {gen_file.name}")

        with open(gen_file) as f:
            data = json.load(f)

        trajectories = data["trajectories"]
        prompt = data.get("prompt_text", "<root>")
        variant = data["prompt_variant"]

        if not trajectories:
            print("  No trajectories")
            continue

        print(f"  {len(trajectories)} trajectories")

        # Load scores
        scores, structures = {}, []
        est_file = result_dir / f"est_{variant}.json"
        if est_file.exists():
            with open(est_file) as f:
                est = json.load(f)
            if est.get("systems"):
                sys = est["systems"][0]
                structures = [s["structure"] for s in sys["structures"]]
                for i, t in enumerate(trajectories):
                    scores[t["text"]] = [s["scores"][i] for s in sys["structures"]]
            print(f"  Loaded {len(structures)} structures")

        # Generate trees
        for name, builder, max_d in [
            ("word_tree", build_word_tree, 10),
            ("phrase_tree", build_phrase_tree, 6),
            ("token_tree", build_token_tree, 15),
        ]:
            tree = builder(trajectories, scores, prompt)
            if tree:
                plot_tree(
                    tree,
                    name.replace("_", " ").title(),
                    output_dir / f"{name}.png",
                    structures,
                    scores,
                )
            elif name == "token_tree":
                print("  Skipping token tree (no token data)")
