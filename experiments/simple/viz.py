"""
Professional visualization for LLM trajectory trees and dynamics.

Creates publication-quality visualizations of:
1. LLM string subtree with probability/deviance encoding
2. Dynamics analysis across structures
3. Core and deviance analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from schemas import (
    CoreEstimationOutput,
    GenerationOutput,
    StructureResult,
    SystemResult,
    TrajectoryRecord,
    VisualizationOutput,
)

# Professional color palettes
COLORS = {
    "primary": "#2E4057",  # Dark blue-gray
    "secondary": "#048A81",  # Teal
    "accent": "#54C6EB",  # Light blue
    "warm": "#F45B69",  # Coral
    "neutral": "#8D99AE",  # Gray-blue
    "background": "#FAFBFC",  # Off-white
    "text": "#1A1A2E",  # Near black
    "grid": "#E5E5E5",  # Light gray
}

# Diverging colormap for deviance (green-yellow-red)
DEVIANCE_COLORS = ["#2E7D32", "#66BB6A", "#FDD835", "#FF8F00", "#D32F2F"]

# Sequential colormap for probability
PROB_COLORS = ["#E3F2FD", "#90CAF9", "#42A5F5", "#1976D2", "#0D47A1"]


# -----------------------------------------------------------------------------
# Input/Output Data Structures
# -----------------------------------------------------------------------------


@dataclass
class VisualizationInput:
    """Input for visualization."""

    output_dir: Path
    param_id: str
    verbose: bool


@dataclass
class VisualizationResult:
    """Output from visualization."""

    viz_output: VisualizationOutput
    tree_path: Path
    summary_path: Path
    dynamics_path: Optional[Path]
    description_path: Path


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def setup_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update(
        {
            # Figure
            "figure.facecolor": COLORS["background"],
            "figure.edgecolor": "none",
            "figure.dpi": 150,
            # Axes
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["grid"],
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": COLORS["text"],
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "medium",
            "axes.labelpad": 8,
            # Ticks
            "xtick.color": COLORS["neutral"],
            "ytick.color": COLORS["neutral"],
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            # Font
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
            # Legend
            "legend.frameon": False,
            "legend.fontsize": 9,
            # Grid
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.5,
            "grid.alpha": 0.7,
        }
    )


class TreeVisualizer:
    """Professional visualization for LLM trajectory trees."""

    def __init__(
        self,
        gen_output: GenerationOutput,
        est_output: Optional[CoreEstimationOutput] = None,
    ):
        self.gen = gen_output
        self.est = est_output
        self.trajectories = gen_output.trajectories
        self.tree = self._build_tree()

    def _build_tree(self) -> dict:
        """Build tree structure from trajectories."""
        root = {
            "token": "",
            "children": {},
            "prob": 1.0,
            "trajectories": list(range(len(self.trajectories))),
            "depth": 0,
        }

        for i, traj in enumerate(self.trajectories):
            text = traj.text
            if "assistant\n" in text:
                continuation = (
                    text.split("assistant\n")[-1].replace("<|im_end|>", "").strip()
                )
            else:
                continuation = text[-50:]

            tokens = list(continuation)
            current = root

            for depth, token in enumerate(tokens, 1):
                if token not in current["children"]:
                    current["children"][token] = {
                        "token": token,
                        "children": {},
                        "prob": 0.0,
                        "trajectories": [],
                        "depth": depth,
                    }
                current = current["children"][token]
                current["prob"] += traj.probability
                if i not in current["trajectories"]:
                    current["trajectories"].append(i)

        return root

    def _get_deviance_for_trajectory(self, traj_idx: int) -> float:
        """Get average deviance for a trajectory across all structures."""
        if self.est is None:
            return 0.0

        deviances = []
        for system in self.est.systems:
            for struct in system.structures:
                if traj_idx < len(struct.scores):
                    deviances.append(abs(struct.scores[traj_idx] - struct.core))

        return np.mean(deviances) if deviances else 0.0

    def _get_node_color(self, node: dict) -> str:
        """Get color for a node based on deviance or probability."""
        if self.est and node["trajectories"]:
            avg_deviance = np.mean(
                [self._get_deviance_for_trajectory(i) for i in node["trajectories"]]
            )
            # Map deviance [0, 0.5] to color index [0, 4]
            idx = min(4, int(avg_deviance * 8))
            return DEVIANCE_COLORS[idx]
        else:
            # Map probability to color
            prob = node["prob"]
            if prob > 0.2:
                return PROB_COLORS[4]
            elif prob > 0.1:
                return PROB_COLORS[3]
            elif prob > 0.05:
                return PROB_COLORS[2]
            elif prob > 0.01:
                return PROB_COLORS[1]
            else:
                return PROB_COLORS[0]

    def plot_tree(
        self,
        output_path: Path,
        max_depth: int = 12,
        min_prob: float = 0.005,
        figsize: tuple = (16, 14),
    ) -> str:
        """Create a professional tree visualization."""
        setup_style()

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("white")

        # Collect all nodes and edges for plotting
        nodes_to_plot = []
        edges_to_plot = []

        def collect_nodes(node, x, y, x_range, parent_pos=None):
            if node["depth"] > max_depth:
                return
            if node["prob"] < min_prob and node["depth"] > 2:
                return

            nodes_to_plot.append(
                {
                    "x": x,
                    "y": y,
                    "prob": node["prob"],
                    "token": node["token"],
                    "depth": node["depth"],
                    "trajectories": node["trajectories"],
                    "color": self._get_node_color(node),
                }
            )

            if parent_pos:
                edges_to_plot.append(
                    {
                        "start": parent_pos,
                        "end": (x, y),
                        "prob": node["prob"],
                    }
                )

            children = sorted(
                node["children"].items(), key=lambda c: c[1]["prob"], reverse=True
            )
            if children:
                n_children = len(children)
                child_x_range = x_range / max(n_children, 1)

                for i, (token, child) in enumerate(children):
                    child_x = x - x_range / 2 + child_x_range * (i + 0.5)
                    child_y = y - 1.2
                    collect_nodes(child, child_x, child_y, child_x_range * 0.85, (x, y))

        # Start collection from root
        collect_nodes(self.tree, 0, 0, 24)

        # Draw edges with gradient thickness
        for edge in edges_to_plot:
            x1, y1 = edge["start"]
            x2, y2 = edge["end"]
            prob = edge["prob"]

            # Line thickness based on probability
            linewidth = max(0.5, min(4, prob * 15))
            alpha = max(0.3, min(0.9, prob * 2 + 0.3))

            # Draw curved connection
            ax.plot(
                [x1, x2],
                [y1, y2],
                color=COLORS["neutral"],
                linewidth=linewidth,
                alpha=alpha,
                solid_capstyle="round",
                zorder=1,
            )

        # Draw nodes
        for node in nodes_to_plot:
            x, y = node["x"], node["y"]
            prob = node["prob"]

            # Node size based on probability
            size = max(80, min(800, prob * 2500))

            # Draw node
            ax.scatter(
                [x],
                [y],
                s=size,
                c=[node["color"]],
                alpha=0.85,
                edgecolors="white",
                linewidth=1.5,
                zorder=2,
            )

            # Add token label for significant nodes
            if prob > 0.03 and node["token"]:
                # Clean up token for display
                display_token = node["token"]
                if display_token == " ":
                    display_token = "␣"
                elif display_token == "\n":
                    display_token = "↵"

                fontsize = max(7, min(11, prob * 40))
                text = ax.text(
                    x,
                    y,
                    display_token,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    fontweight="medium",
                    color="white" if prob > 0.1 else COLORS["text"],
                    zorder=3,
                )
                text.set_path_effects(
                    [path_effects.withStroke(linewidth=2, foreground="white")]
                )

        # Styling
        ax.set_xlim(-13, 13)
        y_min = min(n["y"] for n in nodes_to_plot) - 1 if nodes_to_plot else -15
        ax.set_ylim(y_min, 1.5)

        ax.set_xlabel("Branching Factor", fontsize=11, color=COLORS["text"])
        ax.set_ylabel("Generation Depth (tokens)", fontsize=11, color=COLORS["text"])

        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Title
        title = f"Trajectory Tree: {self.gen.experiment_id}"
        subtitle = f"{len(self.trajectories)} trajectories | {self.gen.total_mass:.1%} probability mass"

        fig.suptitle(
            title, fontsize=16, fontweight="bold", color=COLORS["text"], y=0.96
        )
        ax.set_title(subtitle, fontsize=11, color=COLORS["neutral"], pad=10)

        # Legend
        if self.est:
            legend_elements = [
                mpatches.Patch(facecolor=DEVIANCE_COLORS[0], label="Low deviance"),
                mpatches.Patch(facecolor=DEVIANCE_COLORS[2], label="Medium deviance"),
                mpatches.Patch(facecolor=DEVIANCE_COLORS[4], label="High deviance"),
            ]
            ax.legend(
                handles=legend_elements,
                loc="upper right",
                frameon=True,
                facecolor="white",
                edgecolor=COLORS["grid"],
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(
            output_path, dpi=200, bbox_inches="tight", facecolor=COLORS["background"]
        )
        plt.close()

        return str(output_path)

    def plot_dynamics(
        self,
        output_path: Path,
        figsize: tuple = (16, 12),
    ) -> Optional[str]:
        """Create professional dynamics visualization."""
        if self.est is None:
            return None

        setup_style()

        fig = plt.figure(figsize=figsize, facecolor=COLORS["background"])

        # Create grid layout
        gs = fig.add_gridspec(
            2, 3, hspace=0.35, wspace=0.3, left=0.08, right=0.95, top=0.88, bottom=0.08
        )

        # 1. Structure Scores Heatmap (top-left, spans 2 cols)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_scores_heatmap(ax1)

        # 2. Core Values (top-right)
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_core_values(ax2)

        # 3. Top Trajectories (bottom-left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_top_trajectories(ax3)

        # 4. Deviance Analysis (bottom-middle)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_deviance_analysis(ax4)

        # 5. Best per Structure (bottom-right)
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_best_per_structure(ax5)

        # Main title
        fig.suptitle(
            f"Dynamics Analysis: {self.gen.experiment_id}",
            fontsize=16,
            fontweight="bold",
            color=COLORS["text"],
            y=0.97,
        )

        plt.savefig(
            output_path, dpi=200, bbox_inches="tight", facecolor=COLORS["background"]
        )
        plt.close()

        return str(output_path)

    def _plot_scores_heatmap(self, ax):
        """Plot structure scores as a heatmap."""
        structures = []
        all_scores = []

        for system in self.est.systems:
            for struct in system.structures:
                label = (
                    struct.structure[:25] + "..."
                    if len(struct.structure) > 25
                    else struct.structure
                )
                structures.append(label)
                all_scores.append(struct.scores)

        if not all_scores:
            return

        scores_matrix = np.array(all_scores)

        # Custom colormap
        colors = ["#D32F2F", "#FFEB3B", "#2E7D32"]
        cmap = LinearSegmentedColormap.from_list("custom", colors)

        im = ax.imshow(scores_matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Score", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        ax.set_yticks(range(len(structures)))
        ax.set_yticklabels(structures, fontsize=8)
        ax.set_xlabel("Trajectory Index", fontsize=10)
        ax.set_title(
            "Structure Scores by Trajectory", fontsize=12, fontweight="medium", pad=10
        )

        # Grid
        ax.set_xticks(np.arange(-0.5, scores_matrix.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, scores_matrix.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.5)

    def _plot_core_values(self, ax):
        """Plot core values as horizontal bars."""
        labels = []
        cores = []
        colors = []

        for system in self.est.systems:
            for struct in system.structures:
                short_struct = (
                    struct.structure[:15] + "..."
                    if len(struct.structure) > 15
                    else struct.structure
                )
                labels.append(f"{system.system[:8]}\n{short_struct}")
                cores.append(struct.core)
                # Color based on core value
                if struct.core > 0.6:
                    colors.append(DEVIANCE_COLORS[0])
                elif struct.core > 0.4:
                    colors.append(DEVIANCE_COLORS[2])
                else:
                    colors.append(DEVIANCE_COLORS[4])

        y_pos = np.arange(len(labels))
        bars = ax.barh(
            y_pos, cores, color=colors, alpha=0.85, edgecolor="white", linewidth=1
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Core Value", fontsize=10)
        ax.set_xlim(0, 1)
        ax.axvline(
            x=0.5, color=COLORS["neutral"], linestyle="--", alpha=0.5, linewidth=1
        )
        ax.set_title("Core Values", fontsize=12, fontweight="medium", pad=10)

        # Add value labels
        for bar, val in zip(bars, cores):
            ax.text(
                val + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center",
                fontsize=8,
                color=COLORS["text"],
            )

    def _plot_top_trajectories(self, ax):
        """Plot top trajectories by probability."""
        sorted_trajs = sorted(
            enumerate(self.trajectories), key=lambda x: x[1].probability, reverse=True
        )[:8]

        labels = []
        probs = []

        for idx, traj in sorted_trajs:
            cont = self._get_continuation(idx)[:20]
            labels.append(f'"{cont}"')
            probs.append(traj.probability)

        y_pos = np.arange(len(labels))
        colors = [PROB_COLORS[min(4, int(p * 15))] for p in probs]

        bars = ax.barh(
            y_pos, probs, color=colors, alpha=0.85, edgecolor="white", linewidth=1
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8, family="monospace")
        ax.set_xlabel("Probability", fontsize=10)
        ax.set_title("Top Trajectories", fontsize=12, fontweight="medium", pad=10)
        ax.invert_yaxis()

        # Add probability labels
        for bar, val in zip(bars, probs):
            ax.text(
                val + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}",
                va="center",
                fontsize=8,
                color=COLORS["text"],
            )

    def _plot_deviance_analysis(self, ax):
        """Plot deviance analysis."""
        systems = []
        exp_deviances = []
        var_deviances = []

        for system in self.est.systems:
            for struct in system.structures:
                systems.append(f"{system.system[:6]}\n{struct.structure[:10]}...")
                exp_deviances.append(struct.expected_deviance)
                var_deviances.append(struct.var_deviance)

        x = np.arange(len(systems))
        width = 0.35

        ax.bar(
            x - width / 2,
            exp_deviances,
            width,
            label="Expected",
            color=COLORS["secondary"],
            alpha=0.85,
            edgecolor="white",
        )
        ax.bar(
            x + width / 2,
            var_deviances,
            width,
            label="Variance",
            color=COLORS["warm"],
            alpha=0.85,
            edgecolor="white",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(systems, fontsize=7, rotation=45, ha="right")
        ax.set_ylabel("Deviance", fontsize=10)
        ax.set_title("Deviance Analysis", fontsize=12, fontweight="medium", pad=10)
        ax.legend(fontsize=8, loc="upper right")

    def _plot_best_per_structure(self, ax):
        """Plot best trajectory for each structure."""
        best_trajs = []

        for system in self.est.systems:
            for struct in system.structures:
                if struct.scores:
                    best_idx = int(np.argmax(struct.scores))
                    best_trajs.append(
                        {
                            "structure": struct.structure[:18],
                            "best_idx": best_idx,
                            "score": struct.scores[best_idx],
                            "text": self._get_continuation(best_idx)[:15],
                        }
                    )

        if not best_trajs:
            return

        labels = [f'{t["structure"]}\n> "{t["text"]}"' for t in best_trajs]
        scores = [t["score"] for t in best_trajs]

        y_pos = np.arange(len(labels))
        colors = [
            DEVIANCE_COLORS[0]
            if s > 0.7
            else DEVIANCE_COLORS[2]
            if s > 0.4
            else DEVIANCE_COLORS[4]
            for s in scores
        ]

        bars = ax.barh(
            y_pos, scores, color=colors, alpha=0.85, edgecolor="white", linewidth=1
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Score", fontsize=10)
        ax.set_xlim(0, 1.1)
        ax.set_title(
            "Best Trajectory per Structure", fontsize=12, fontweight="medium", pad=10
        )

        for bar, val in zip(bars, scores):
            ax.text(
                val + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center",
                fontsize=8,
                color=COLORS["text"],
            )

    def _get_continuation(self, traj_idx: int) -> str:
        """Get continuation text for a trajectory."""
        if traj_idx >= len(self.trajectories):
            return ""
        text = self.trajectories[traj_idx].text
        if "assistant\n" in text:
            return text.split("assistant\n")[-1].replace("<|im_end|>", "").strip()
        return text[-30:]

    def plot_summary(
        self,
        output_path: Path,
        figsize: tuple = (12, 8),
    ) -> str:
        """Create a single-page summary visualization."""
        setup_style()

        fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor=COLORS["background"])

        # Left: Probability distribution
        ax1 = axes[0]
        probs = [t.probability for t in self.trajectories]
        sorted_probs = sorted(probs, reverse=True)

        # Cumulative probability
        cumsum = np.cumsum(sorted_probs)
        x = range(1, len(sorted_probs) + 1)

        ax1.fill_between(x, cumsum, alpha=0.3, color=COLORS["secondary"])
        ax1.plot(x, cumsum, color=COLORS["secondary"], linewidth=2, label="Cumulative")
        ax1.bar(x, sorted_probs, color=COLORS["primary"], alpha=0.7, label="Individual")

        ax1.set_xlabel("Trajectory Rank", fontsize=11)
        ax1.set_ylabel("Probability", fontsize=11)
        ax1.set_title("Probability Distribution", fontsize=13, fontweight="medium")
        ax1.legend(fontsize=9)
        ax1.set_xlim(0, len(sorted_probs) + 1)

        # Right: Top completions
        ax2 = axes[1]
        sorted_trajs = sorted(
            enumerate(self.trajectories), key=lambda x: x[1].probability, reverse=True
        )[:10]

        labels = []
        probs = []
        for idx, traj in sorted_trajs:
            cont = self._get_continuation(idx)
            labels.append(f'"{cont[:25]}"')
            probs.append(traj.probability)

        y_pos = np.arange(len(labels))
        colors = plt.cm.Blues(np.linspace(0.9, 0.4, len(probs)))

        ax2.barh(y_pos, probs, color=colors, edgecolor="white", linewidth=1)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels, fontsize=9, family="monospace")
        ax2.set_xlabel("Probability", fontsize=11)
        ax2.set_title("Top Completions", fontsize=13, fontweight="medium")
        ax2.invert_yaxis()

        for i, (y, p) in enumerate(zip(y_pos, probs)):
            ax2.text(p + 0.005, y, f"{p:.1%}", va="center", fontsize=9)

        fig.suptitle(
            f"Summary: {self.gen.experiment_id}",
            fontsize=16,
            fontweight="bold",
            color=COLORS["text"],
            y=0.98,
        )

        plt.tight_layout()
        plt.savefig(
            output_path, dpi=200, bbox_inches="tight", facecolor=COLORS["background"]
        )
        plt.close()

        return str(output_path)

    def generate_description(self) -> str:
        """Generate markdown description of the visualization."""
        lines = [
            f"# Trajectory Analysis: {self.gen.experiment_id}",
            "",
            "## Overview",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total trajectories | {len(self.trajectories)} |",
            f"| Probability mass | {self.gen.total_mass:.2%} |",
            f"| Model | {self.gen.model} |",
            "",
            "## Prompt",
            "",
            f"> {self.gen.prompt_text}",
            "",
            "## Top Trajectories",
            "",
            "| Rank | Probability | Completion |",
            "|------|-------------|------------|",
        ]

        sorted_trajs = sorted(
            self.trajectories, key=lambda t: t.probability, reverse=True
        )[:10]
        for i, traj in enumerate(sorted_trajs, 1):
            cont = self._get_continuation(self.trajectories.index(traj))
            lines.append(f"| {i} | {traj.probability:.2%} | {cont} |")

        if self.est:
            lines.extend(
                [
                    "",
                    "## Structure Analysis",
                    "",
                ]
            )
            for system in self.est.systems:
                lines.append(f"### {system.system}")
                lines.append("")
                lines.append("| Structure | Core | Deviance |")
                lines.append("|-----------|------|----------|")
                for struct in system.structures:
                    lines.append(
                        f"| {struct.structure[:40]} | {struct.core:.3f} | {struct.expected_deviance:.3f} |"
                    )
                lines.append("")

        return "\n".join(lines)


def load_outputs(
    output_dir: Path, param_id: str
) -> tuple[list[GenerationOutput], list[CoreEstimationOutput]]:
    """Load generation and estimation outputs for a param_id."""
    gen_outputs = []
    est_outputs = []

    for f in output_dir.glob(f"{param_id}_gen_*.json"):
        with open(f) as fp:
            data = json.load(fp)
            trajectories = [TrajectoryRecord(**t) for t in data.get("trajectories", [])]
            gen_output = GenerationOutput(
                param_id=data["param_id"],
                experiment_id=data["experiment_id"],
                prompt_variant=data["prompt_variant"],
                prompt_text=data["prompt_text"],
                model=data["model"],
                timestamp=data["timestamp"],
                min_prob_mass=data["min_prob_mass"],
                total_mass=data["total_mass"],
                num_trajectories=data["num_trajectories"],
                trajectories=trajectories,
            )
            gen_outputs.append(gen_output)

    for f in output_dir.glob(f"{param_id}_est_*.json"):
        with open(f) as fp:
            data = json.load(fp)
            systems = []
            for sys_data in data.get("systems", []):
                structures = [
                    StructureResult(**s) for s in sys_data.get("structures", [])
                ]
                systems.append(
                    SystemResult(
                        system=sys_data["system"],
                        structures=structures,
                        aggregate_core=sys_data["aggregate_core"],
                        aggregate_deviance=sys_data["aggregate_deviance"],
                    )
                )
            est_output = CoreEstimationOutput(
                param_id=data["param_id"],
                experiment_id=data["experiment_id"],
                timestamp=data["timestamp"],
                judge_model=data["judge_model"],
                prompt_variant=data["prompt_variant"],
                prompt_text=data["prompt_text"],
                num_trajectories=data["num_trajectories"],
                total_mass=data["total_mass"],
                systems=systems,
            )
            est_outputs.append(est_output)

    return gen_outputs, est_outputs


def merge_generation_outputs(outputs: list[GenerationOutput]) -> GenerationOutput:
    """Merge multiple generation outputs into one."""
    if not outputs:
        raise ValueError("No outputs to merge")

    merged = outputs[0]
    seen_texts = set()
    all_trajectories = []

    for output in outputs:
        for traj in output.trajectories:
            if traj.text not in seen_texts:
                seen_texts.add(traj.text)
                all_trajectories.append(traj)

    all_trajectories.sort(key=lambda t: t.probability, reverse=True)

    return GenerationOutput(
        param_id=merged.param_id,
        experiment_id=merged.experiment_id,
        prompt_variant=merged.prompt_variant,
        prompt_text=merged.prompt_text,
        model=merged.model,
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        min_prob_mass=merged.min_prob_mass,
        total_mass=sum(t.probability for t in all_trajectories),
        num_trajectories=len(all_trajectories),
        trajectories=all_trajectories,
    )


# -----------------------------------------------------------------------------
# Core Logic
# -----------------------------------------------------------------------------


def visualize(inp: VisualizationInput) -> VisualizationResult:
    """Main visualization logic."""
    viz_dir = inp.output_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    if inp.verbose:
        print(f"Loading outputs from: {inp.output_dir}")
        print(f"Param ID: {inp.param_id}")

    gen_outputs, est_outputs = load_outputs(inp.output_dir, inp.param_id)

    if not gen_outputs:
        raise FileNotFoundError(f"No generation outputs found for {inp.param_id}")

    if inp.verbose:
        print(f"Found {len(gen_outputs)} generation output(s)")
        print(f"Found {len(est_outputs)} estimation output(s)")

    merged_gen = merge_generation_outputs(gen_outputs)

    if inp.verbose:
        print(f"Merged into {merged_gen.num_trajectories} unique trajectories")

    est_output = est_outputs[0] if est_outputs else None
    visualizer = TreeVisualizer(merged_gen, est_output)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate all visualizations
    tree_path = viz_dir / f"{inp.param_id}_tree_{timestamp}.png"
    visualizer.plot_tree(tree_path)
    if inp.verbose:
        print(f"Saved: {tree_path}")

    summary_path = viz_dir / f"{inp.param_id}_summary_{timestamp}.png"
    visualizer.plot_summary(summary_path)
    if inp.verbose:
        print(f"Saved: {summary_path}")

    dynamics_path = None
    if est_output:
        dynamics_path = viz_dir / f"{inp.param_id}_dynamics_{timestamp}.png"
        visualizer.plot_dynamics(dynamics_path)
        if inp.verbose:
            print(f"Saved: {dynamics_path}")

    description = visualizer.generate_description()
    desc_path = viz_dir / f"{inp.param_id}_description_{timestamp}.md"
    with open(desc_path, "w") as f:
        f.write(description)
    if inp.verbose:
        print(f"Saved: {desc_path}")
        print()
        print(description)

    viz_output = VisualizationOutput(
        param_id=inp.param_id,
        experiment_id=merged_gen.experiment_id,
        timestamp=timestamp,
        prompt_variant=merged_gen.prompt_variant,
        tree_image_path=str(tree_path),
        dynamics_image_path=str(dynamics_path) if dynamics_path else None,
        description=description,
    )

    return VisualizationResult(
        viz_output=viz_output,
        tree_path=tree_path,
        summary_path=summary_path,
        dynamics_path=dynamics_path,
        description_path=desc_path,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize trajectory trees and dynamics"
    )
    parser.add_argument(
        "--trial",
        type=str,
        default="synthetic_test",
        help="Trial name (looks for outputs in outputs/simple/{trial}_{param_id}/)",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path(__file__).parent.parent.parent / "outputs" / "simple",
        help="Base output directory",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Direct path to experiment output directory",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    return parser.parse_args()


def input_from_args(args: argparse.Namespace) -> VisualizationInput:
    """Load input from command line arguments."""
    if args.experiment_dir:
        experiment_dir = args.experiment_dir
        param_id = experiment_dir.name.split("_")[-1]
    else:
        matching_dirs = list(args.output_base.glob(f"{args.trial}_*"))
        if not matching_dirs:
            raise FileNotFoundError(
                f"No experiment directories found for trial: {args.trial}"
            )
        experiment_dir = max(matching_dirs, key=lambda d: d.stat().st_mtime)
        param_id = experiment_dir.name.split("_")[-1]

    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    return VisualizationInput(
        output_dir=experiment_dir,
        param_id=param_id,
        verbose=not args.quiet,
    )


def save_output(args: argparse.Namespace, output: VisualizationResult) -> None:
    """Save visualization output record to JSON file."""
    viz_record_path = (
        output.tree_path.parent
        / f"{output.viz_output.param_id}_viz_{output.viz_output.timestamp}.json"
    )
    with open(viz_record_path, "w") as f:
        json.dump(asdict(output.viz_output), f, indent=2)


def print_output(args: argparse.Namespace, output: VisualizationResult) -> None:
    """Print output summary (already printed during visualization)."""
    pass


def main() -> int:
    args = get_args()
    inp: VisualizationInput = input_from_args(args)
    output: VisualizationResult = visualize(inp)

    save_output(args, output)
    print_output(args, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
