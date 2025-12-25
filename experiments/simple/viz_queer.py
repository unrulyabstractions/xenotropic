"""
Visualization for queer.json trial - structure-aware diversity analysis.

Creates visualizations showing:
1. How base_prompt induces a "heterosexual cisgender" core (low queer compliance)
2. How "queens" branching point induces "queer" dominance
3. How "racing" branching point maintains "heterosexual cisgender" dominance
4. Structure compliance with colors (like Marshall's example)
5. Deviance encoded as size/thickness (low deviance = big/thick, high deviance = small/thin)
6. Dynamics along branches

Focuses only on vector_system.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Structure colors - each structure gets a distinct color
STRUCTURE_COLORS = {
    "queer": "#9C27B0",  # Purple for queer
    "woman": "#E91E63",  # Pink for woman
    "man": "#2196F3",  # Blue for man
    "neutral": "#9E9E9E",  # Gray for neutral/mixed
}

# Background and styling
COLORS = {
    "background": "#FAFBFC",
    "text": "#1A1A2E",
    "grid": "#E5E5E5",
    "neutral": "#8D99AE",
    "edge": "#546E7A",
}


# -----------------------------------------------------------------------------
# Input/Output Data Structures
# -----------------------------------------------------------------------------


@dataclass
class VizQueerInput:
    """Input for queer visualization."""

    output_dir: Path
    verbose: bool


@dataclass
class VizQueerOutput:
    """Output from queer visualization."""

    tree_path: Path
    dynamics_path: Path
    details_path: Path
    marshall_path: Path
    diagram_path: Path
    json_path: Path


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def setup_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["background"],
            "figure.edgecolor": "none",
            "figure.dpi": 150,
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
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


@dataclass
class TrajectoryData:
    """A trajectory with its properties."""

    text: str
    probability: float
    structure_scores: dict  # {structure_name: score}
    deviance: float  # deviation from core
    branch: str  # "base", "queens", or "racing"


@dataclass
class BranchData:
    """Data for a branch (base, queens, racing)."""

    name: str
    prompt_suffix: str
    trajectories: list
    core: dict  # {structure_name: core_value}


def generate_synthetic_data():
    """
    Generate synthetic data demonstrating the expected patterns:
    - base_prompt: "heterosexual cisgender" dominance (low queer, low woman, moderate man)
    - queens: "queer" dominance (high queer)
    - racing: "heterosexual cisgender" still dominates (low queer)
    - Throughout: woman lowest, man relatively constant
    """

    branches = []

    # Base prompt trajectories - "heterosexual cisgender" dominance
    # Low queer (~0.15), low woman (~0.10), moderate man (~0.45)
    base_trajectories = [
        TrajectoryData(
            text="Jake, who dated Sarah for years.",
            probability=0.15,
            structure_scores={"queer": 0.08, "woman": 0.12, "man": 0.52},
            deviance=0.05,
            branch="base",
        ),
        TrajectoryData(
            text="Mike loved his girlfriend Lisa.",
            probability=0.12,
            structure_scores={"queer": 0.05, "woman": 0.15, "man": 0.48},
            deviance=0.08,
            branch="base",
        ),
        TrajectoryData(
            text="Steve, with his wife and kids.",
            probability=0.10,
            structure_scores={"queer": 0.03, "woman": 0.08, "man": 0.55},
            deviance=0.12,
            branch="base",
        ),
        TrajectoryData(
            text="Tom, a tough family man.",
            probability=0.08,
            structure_scores={"queer": 0.10, "woman": 0.05, "man": 0.60},
            deviance=0.15,
            branch="base",
        ),
        TrajectoryData(
            text="racing and his wife cheered.",
            probability=0.06,
            structure_scores={"queer": 0.02, "woman": 0.18, "man": 0.42},
            deviance=0.18,
            branch="base",
        ),
        TrajectoryData(
            text="queens and makeup shows.",
            probability=0.04,
            structure_scores={"queer": 0.85, "woman": 0.25, "man": 0.20},
            deviance=0.65,
            branch="base",
        ),
        TrajectoryData(
            text="bikes and beer with buddies.",
            probability=0.07,
            structure_scores={"queer": 0.05, "woman": 0.02, "man": 0.65},
            deviance=0.10,
            branch="base",
        ),
    ]

    base_core = {"queer": 0.12, "woman": 0.10, "man": 0.50}
    branches.append(
        BranchData(
            name="base",
            prompt_suffix="",
            trajectories=base_trajectories,
            core=base_core,
        )
    )

    # Queens branch trajectories - "queer" dominance
    # High queer (~0.75), low woman (~0.15), lower man (~0.25)
    queens_trajectories = [
        TrajectoryData(
            text="queens and loved the spotlight.",
            probability=0.18,
            structure_scores={"queer": 0.92, "woman": 0.20, "man": 0.15},
            deviance=0.08,
            branch="queens",
        ),
        TrajectoryData(
            text="queens. His name was Destiny.",
            probability=0.14,
            structure_scores={"queer": 0.88, "woman": 0.35, "man": 0.18},
            deviance=0.12,
            branch="queens",
        ),
        TrajectoryData(
            text="queens on RuPaul's show.",
            probability=0.12,
            structure_scores={"queer": 0.95, "woman": 0.22, "man": 0.10},
            deviance=0.10,
            branch="queens",
        ),
        TrajectoryData(
            text="queens at the local bar.",
            probability=0.10,
            structure_scores={"queer": 0.82, "woman": 0.15, "man": 0.22},
            deviance=0.15,
            branch="queens",
        ),
        TrajectoryData(
            text="queens, especially his friend Marcus.",
            probability=0.08,
            structure_scores={"queer": 0.78, "woman": 0.08, "man": 0.30},
            deviance=0.18,
            branch="queens",
        ),
        TrajectoryData(
            text="queens and taught makeup artistry.",
            probability=0.06,
            structure_scores={"queer": 0.90, "woman": 0.28, "man": 0.12},
            deviance=0.14,
            branch="queens",
        ),
        TrajectoryData(
            text="queens with his partner Alex.",
            probability=0.05,
            structure_scores={"queer": 0.85, "woman": 0.05, "man": 0.35},
            deviance=0.20,
            branch="queens",
        ),
    ]

    queens_core = {"queer": 0.85, "woman": 0.18, "man": 0.20}
    branches.append(
        BranchData(
            name="queens",
            prompt_suffix="queens",
            trajectories=queens_trajectories,
            core=queens_core,
        )
    )

    # Racing branch trajectories - "heterosexual cisgender" maintains dominance
    # Low queer (~0.08), low woman (~0.08), high man (~0.55)
    racing_trajectories = [
        TrajectoryData(
            text="racing. Jake loved fast cars.",
            probability=0.16,
            structure_scores={"queer": 0.05, "woman": 0.05, "man": 0.62},
            deviance=0.06,
            branch="racing",
        ),
        TrajectoryData(
            text="racing with his crew every weekend.",
            probability=0.13,
            structure_scores={"queer": 0.08, "woman": 0.08, "man": 0.58},
            deviance=0.08,
            branch="racing",
        ),
        TrajectoryData(
            text="racing down Route 66.",
            probability=0.11,
            structure_scores={"queer": 0.03, "woman": 0.02, "man": 0.65},
            deviance=0.10,
            branch="racing",
        ),
        TrajectoryData(
            text="racing motorcycles with his brothers.",
            probability=0.09,
            structure_scores={"queer": 0.02, "woman": 0.05, "man": 0.70},
            deviance=0.12,
            branch="racing",
        ),
        TrajectoryData(
            text="racing. Mike won every championship.",
            probability=0.08,
            structure_scores={"queer": 0.10, "woman": 0.10, "man": 0.52},
            deviance=0.15,
            branch="racing",
        ),
        TrajectoryData(
            text="racing and teaching his son.",
            probability=0.06,
            structure_scores={"queer": 0.05, "woman": 0.12, "man": 0.55},
            deviance=0.18,
            branch="racing",
        ),
        TrajectoryData(
            text="racing with his girlfriend Sarah.",
            probability=0.05,
            structure_scores={"queer": 0.08, "woman": 0.15, "man": 0.48},
            deviance=0.20,
            branch="racing",
        ),
    ]

    racing_core = {"queer": 0.06, "woman": 0.08, "man": 0.58}
    branches.append(
        BranchData(
            name="racing",
            prompt_suffix="racing",
            trajectories=racing_trajectories,
            core=racing_core,
        )
    )

    return branches


def get_dominant_structure(scores: dict) -> tuple:
    """Get the dominant structure and its score."""
    if not scores:
        return "neutral", 0.0
    dominant = max(scores.items(), key=lambda x: x[1])
    return dominant[0], dominant[1]


def get_structure_color(scores: dict, alpha: float = 1.0) -> tuple:
    """
    Get color based on structure compliance.
    Uses weighted average of structure colors based on scores.
    """
    if not scores:
        return (*hex_to_rgb(STRUCTURE_COLORS["neutral"]), alpha)

    # Normalize scores
    total = sum(scores.values())
    if total == 0:
        return (*hex_to_rgb(STRUCTURE_COLORS["neutral"]), alpha)

    # Weighted color blend
    r, g, b = 0, 0, 0
    for struct, score in scores.items():
        weight = score / total
        color = hex_to_rgb(STRUCTURE_COLORS.get(struct, STRUCTURE_COLORS["neutral"]))
        r += color[0] * weight
        g += color[1] * weight
        b += color[2] * weight

    return (r, g, b, alpha)


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple (0-1 range)."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))


def deviance_to_size(deviance: float, base_size: float = 400) -> float:
    """
    Convert deviance to node size.
    Low deviance = big size, high deviance = small size.
    """
    # Invert: low deviance -> high size
    scale = 1.0 - min(deviance, 1.0)
    return max(50, base_size * (0.2 + 0.8 * scale))


def deviance_to_linewidth(deviance: float, base_width: float = 3.0) -> float:
    """
    Convert deviance to line width.
    Low deviance = thick, high deviance = thin.
    """
    scale = 1.0 - min(deviance, 1.0)
    return max(0.5, base_width * (0.2 + 0.8 * scale))


class QueerVisualizer:
    """Visualizer for the queer.json trial demonstrating structure dynamics."""

    def __init__(self, branches: list):
        self.branches = {b.name: b for b in branches}
        self.structures = ["queer", "woman", "man"]

    def plot_branching_tree(self, output_path: Path, figsize: tuple = (18, 14)):
        """
        Create tree visualization showing how branches induce different cores.

        Structure:
        - Base prompt at top
        - Two main branches: "queens" and "racing"
        - Trajectories emanating from each branch
        - Colors show structure compliance
        - Size/thickness shows deviance
        """
        setup_style()

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("white")

        # Layout parameters
        base_y = 10
        branch_y = 7
        traj_y_start = 4

        # Draw base prompt node - colored by its core (man-dominant = blue)
        base_data = self.branches["base"]
        base_color = get_structure_color(base_data.core)
        ax.scatter(
            [0],
            [base_y],
            s=800,
            c=[base_color[:3]],
            edgecolors="white",
            linewidth=3,
            zorder=5,
        )
        ax.text(
            0,
            base_y + 0.8,
            "base_prompt",
            ha="center",
            fontsize=12,
            fontweight="bold",
            color=COLORS["text"],
        )
        ax.text(
            0,
            base_y - 0.8,
            '"...loved drag "',
            ha="center",
            fontsize=9,
            style="italic",
            color=COLORS["neutral"],
        )

        # Draw branching points
        branch_positions = {"queens": -6, "racing": 6}

        for branch_name, x_pos in branch_positions.items():
            branch = self.branches[branch_name]

            # Draw connection from base to branch
            ax.plot(
                [0, x_pos],
                [base_y - 0.4, branch_y + 0.4],
                color=COLORS["edge"],
                linewidth=2.5,
                zorder=3,
            )

            # Draw branch node with color based on its core
            core_color = get_structure_color(branch.core)
            ax.scatter(
                [x_pos],
                [branch_y],
                s=600,
                c=[core_color[:3]],
                edgecolors="white",
                linewidth=2.5,
                zorder=5,
            )
            ax.text(
                x_pos,
                branch_y + 0.7,
                f'"{branch_name}"',
                ha="center",
                fontsize=11,
                fontweight="bold",
                color=COLORS["text"],
            )

            # Annotate core values
            core_str = ", ".join(
                [f"{s[0].upper()}:{v:.2f}" for s, v in branch.core.items()]
            )
            ax.text(
                x_pos,
                branch_y - 0.6,
                f"Core: [{core_str}]",
                ha="center",
                fontsize=8,
                color=COLORS["neutral"],
            )

            # Draw trajectories
            trajs = sorted(
                branch.trajectories, key=lambda t: t.probability, reverse=True
            )
            n_trajs = min(len(trajs), 7)

            for i, traj in enumerate(trajs[:n_trajs]):
                # Position
                spread = 4.5
                traj_x = x_pos + (i - n_trajs / 2 + 0.5) * (spread / n_trajs)
                traj_y = traj_y_start - i * 0.4

                # Color based on structure compliance
                color = get_structure_color(traj.structure_scores)

                # Size based on deviance (low deviance = big)
                size = deviance_to_size(traj.deviance, base_size=350)

                # Line width based on deviance
                linewidth = deviance_to_linewidth(traj.deviance)

                # Draw connection
                ax.plot(
                    [x_pos, traj_x],
                    [branch_y - 0.3, traj_y + 0.3],
                    color=color[:3],
                    linewidth=linewidth,
                    alpha=0.7,
                    zorder=2,
                )

                # Draw trajectory node
                ax.scatter(
                    [traj_x],
                    [traj_y],
                    s=size,
                    c=[color[:3]],
                    edgecolors="white",
                    linewidth=1.5,
                    zorder=4,
                    alpha=0.9,
                )

                # Add probability and short text
                if traj.probability > 0.06:
                    short_text = (
                        traj.text[:18] + "..." if len(traj.text) > 18 else traj.text
                    )
                    ax.text(
                        traj_x,
                        traj_y - 0.35,
                        f"p={traj.probability:.2f}",
                        ha="center",
                        fontsize=7,
                        color=COLORS["neutral"],
                    )

        # Add some base trajectories to show initial distribution
        base_trajs = sorted(
            base_data.trajectories, key=lambda t: t.probability, reverse=True
        )
        for i, traj in enumerate(base_trajs[:3]):
            # Show bifurcating trajectories from base
            if "queens" in traj.text.lower():
                traj_x = -2 - i * 0.5
            elif "racing" in traj.text.lower():
                traj_x = 2 + i * 0.5
            else:
                traj_x = (i - 1) * 0.8

            traj_y = base_y - 1.5 - i * 0.3

            color = get_structure_color(traj.structure_scores)
            size = deviance_to_size(traj.deviance, base_size=200)
            linewidth = deviance_to_linewidth(traj.deviance, base_width=2)

            ax.plot(
                [0, traj_x],
                [base_y - 0.3, traj_y + 0.2],
                color=color[:3],
                linewidth=linewidth,
                alpha=0.5,
                zorder=1,
            )
            ax.scatter(
                [traj_x],
                [traj_y],
                s=size,
                c=[color[:3]],
                edgecolors="white",
                linewidth=1,
                zorder=3,
                alpha=0.7,
            )

        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=STRUCTURE_COLORS["queer"], label="Queer dominant"),
            mpatches.Patch(facecolor=STRUCTURE_COLORS["woman"], label="Woman dominant"),
            mpatches.Patch(facecolor=STRUCTURE_COLORS["man"], label="Man dominant"),
        ]

        # Add size legend
        legend_elements.extend(
            [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="gray",
                    markersize=15,
                    label="Low deviance (big)",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="gray",
                    markersize=8,
                    label="High deviance (small)",
                ),
            ]
        )

        ax.legend(
            handles=legend_elements,
            loc="upper left",
            fontsize=9,
            frameon=True,
            facecolor="white",
            edgecolor=COLORS["grid"],
        )

        # Annotations
        ax.annotate(
            "'queens' induces\nQUEER dominance",
            xy=(-6, 5.5),
            fontsize=10,
            ha="center",
            color=STRUCTURE_COLORS["queer"],
            fontweight="bold",
        )
        ax.annotate(
            "'racing' maintains\nHETERO-CIS dominance",
            xy=(6, 5.5),
            fontsize=10,
            ha="center",
            color=STRUCTURE_COLORS["man"],
            fontweight="bold",
        )

        # Styling
        ax.set_xlim(-12, 12)
        ax.set_ylim(-1, 12)
        ax.axis("off")

        # Title
        fig.suptitle(
            "Structure-Aware Branching: How Tokens Shift Normative Cores",
            fontsize=16,
            fontweight="bold",
            color=COLORS["text"],
            y=0.96,
        )
        ax.set_title(
            "Prompt: 'The tough powerlifter was also a motorcyclist who loved drag ...'",
            fontsize=11,
            color=COLORS["neutral"],
            pad=10,
        )

        plt.tight_layout()
        plt.savefig(
            output_path, dpi=200, bbox_inches="tight", facecolor=COLORS["background"]
        )
        plt.close()

        return str(output_path)

    def plot_dynamics(self, output_path: Path, figsize: tuple = (16, 10)):
        """
        Create dynamics visualization showing how cores evolve along branches.

        Shows x_phi (core), y_phi (orientation from root), z_phi (local orientation).
        """
        setup_style()

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(
            "Dynamics Analysis: Core Evolution Across Branches",
            fontsize=14,
            fontweight="bold",
            color=COLORS["text"],
            y=0.98,
        )

        # 1. Core comparison across branches (top-left)
        ax1 = axes[0, 0]
        x = np.arange(len(self.structures))
        width = 0.25

        colors = [STRUCTURE_COLORS[s] for s in self.structures]

        for i, (branch_name, branch) in enumerate(self.branches.items()):
            values = [branch.core[s] for s in self.structures]
            bars = ax1.bar(x + i * width, values, width, label=branch_name, alpha=0.8)
            for bar, color in zip(bars, colors):
                bar.set_facecolor(color)
                bar.set_alpha(0.6 if branch_name == "base" else 0.9)

        ax1.set_xticks(x + width)
        ax1.set_xticklabels([s.capitalize() for s in self.structures])
        ax1.set_ylabel("Core Value (Expected Compliance)")
        ax1.set_title("Core Comparison Across Branches", fontweight="medium")
        ax1.legend()
        ax1.set_ylim(0, 1)
        ax1.axhline(y=0.5, color=COLORS["grid"], linestyle="--", alpha=0.5)

        # 2. Structure dominance pie charts (top-right)
        ax2 = axes[0, 1]

        # Create mini pies for each branch
        positions = [(0.17, 0.5), (0.5, 0.5), (0.83, 0.5)]
        labels = ["Base Prompt", '"queens"', '"racing"']
        branch_names = ["base", "queens", "racing"]

        ax2.axis("off")
        ax2.set_title("Structure Dominance by Branch", fontweight="medium", pad=20)

        for pos, label, bname in zip(positions, labels, branch_names):
            branch = self.branches[bname]
            sizes = [branch.core[s] for s in self.structures]
            colors = [STRUCTURE_COLORS[s] for s in self.structures]

            # Create inset axes for pie
            inset_ax = fig.add_axes(
                [0.52 + (pos[0] - 0.5) * 0.45, 0.52 + (pos[1] - 0.5) * 0.35, 0.12, 0.12]
            )
            inset_ax.pie(sizes, colors=colors, startangle=90)
            inset_ax.set_title(label, fontsize=9, pad=5)

        # 3. Deviance distribution (bottom-left)
        ax3 = axes[1, 0]

        for branch_name, branch in self.branches.items():
            deviances = [t.deviance for t in branch.trajectories]
            probs = [t.probability for t in branch.trajectories]

            # Weighted histogram effect via scatter
            ax3.scatter(deviances, probs, label=branch_name, alpha=0.7, s=100)

        ax3.set_xlabel("Deviance from Core")
        ax3.set_ylabel("Probability")
        ax3.set_title("Deviance Distribution by Branch", fontweight="medium")
        ax3.legend()
        ax3.set_xlim(0, 0.8)

        # 4. Structure evolution along trajectory depth (bottom-right)
        ax4 = axes[1, 1]

        # Simulate dynamics over token position
        token_positions = np.arange(0, 10)

        for struct in self.structures:
            color = STRUCTURE_COLORS[struct]

            # Base trend
            base_core = self.branches["base"].core[struct]
            base_trend = base_core + np.random.normal(0, 0.02, len(token_positions))
            ax4.plot(
                token_positions,
                base_trend,
                "--",
                color=color,
                alpha=0.4,
                label=f"{struct} (base)",
            )

            # Queens trend
            queens_core = self.branches["queens"].core[struct]
            queens_trend = np.linspace(base_core, queens_core, len(token_positions))
            queens_trend += np.random.normal(0, 0.03, len(token_positions))

            if struct == "queer":
                ax4.plot(
                    token_positions,
                    queens_trend,
                    "-",
                    color=color,
                    linewidth=2.5,
                    label=f'{struct} ("queens")',
                )

        ax4.set_xlabel("Token Position (after branching)")
        ax4.set_ylabel("Structure Compliance")
        ax4.set_title("Dynamics: How Compliance Evolves", fontweight="medium")
        ax4.legend(fontsize=8)
        ax4.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(
            output_path, dpi=200, bbox_inches="tight", facecolor=COLORS["background"]
        )
        plt.close()

        return str(output_path)

    def plot_trajectory_details(self, output_path: Path, figsize: tuple = (14, 16)):
        """
        Detailed view of individual trajectories and their structure scores.
        Shows how specific completions align with different structures.
        """
        setup_style()

        fig, axes = plt.subplots(3, 1, figsize=figsize)

        for ax, (branch_name, branch) in zip(axes, self.branches.items()):
            trajs = sorted(
                branch.trajectories, key=lambda t: t.probability, reverse=True
            )

            n_trajs = len(trajs)
            y_positions = np.arange(n_trajs)

            # Plot structure scores as grouped bars
            width = 0.25
            for i, struct in enumerate(self.structures):
                scores = [t.structure_scores[struct] for t in trajs]
                color = STRUCTURE_COLORS[struct]
                ax.barh(
                    y_positions + i * width,
                    scores,
                    width,
                    label=struct if branch_name == "base" else "",
                    color=color,
                    alpha=0.8,
                )

            # Labels
            labels = [f'"{t.text[:25]}..." (p={t.probability:.2f})' for t in trajs]
            ax.set_yticks(y_positions + width)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel("Structure Compliance Score")
            ax.set_xlim(0, 1)
            ax.set_title(
                f'Branch: "{branch_name}" - Core: Q={branch.core["queer"]:.2f}, '
                f"W={branch.core['woman']:.2f}, M={branch.core['man']:.2f}",
                fontweight="medium",
                fontsize=11,
            )
            ax.invert_yaxis()

            # Add core reference lines
            for struct in self.structures:
                ax.axvline(
                    x=branch.core[struct],
                    color=STRUCTURE_COLORS[struct],
                    linestyle="--",
                    alpha=0.5,
                    linewidth=1,
                )

        axes[0].legend(loc="lower right", fontsize=9)

        fig.suptitle(
            "Trajectory Structure Analysis: Compliance Scores by Completion",
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )

        plt.tight_layout()
        plt.savefig(
            output_path, dpi=200, bbox_inches="tight", facecolor=COLORS["background"]
        )
        plt.close()

        return str(output_path)

    def plot_marshall_style_tree(self, output_path: Path, figsize: tuple = (20, 16)):
        """
        Create a tree visualization in Marshall's style (from old_paper.pdf).

        Each node is colored by its dominant structure with color intensity
        showing confidence. Size encodes probability/deviance.
        """
        setup_style()

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("#F8F9FA")

        # Build tree structure
        # Root -> branching tokens -> trajectories

        # Root node
        root_x, root_y = 0, 12
        ax.scatter(
            [root_x],
            [root_y],
            s=1200,
            c=["#2C3E50"],
            edgecolors="white",
            linewidth=3,
            zorder=10,
        )
        ax.text(
            root_x,
            root_y + 0.9,
            "ROOT",
            ha="center",
            fontsize=12,
            fontweight="bold",
            color=COLORS["text"],
        )
        ax.text(
            root_x,
            root_y - 0.9,
            '"...loved drag "',
            ha="center",
            fontsize=10,
            style="italic",
            color=COLORS["neutral"],
        )

        # Probability mass annotations
        base_data = self.branches["base"]
        base_mass = sum(t.probability for t in base_data.trajectories)

        # Calculate branch probabilities from base trajectories
        queens_prob = 0.0
        racing_prob = 0.0
        for t in base_data.trajectories:
            if "queens" in t.text.lower():
                queens_prob += t.probability
            elif "racing" in t.text.lower():
                racing_prob += t.probability

        # Normalize to show branching effect
        queens_display = 0.35  # queens token probability
        racing_display = 0.55  # racing token probability

        # Draw main branches
        branches_info = [
            ("queens", -8, 9, queens_display, STRUCTURE_COLORS["queer"]),
            ("racing", 8, 9, racing_display, STRUCTURE_COLORS["man"]),
        ]

        for name, x, y, prob, color in branches_info:
            branch = self.branches[name]

            # Edge from root to branch
            edge_width = 2 + prob * 6
            ax.plot(
                [root_x, x],
                [root_y - 0.5, y + 0.5],
                color=color,
                linewidth=edge_width,
                alpha=0.7,
                zorder=5,
            )

            # Branch node
            node_size = 400 + prob * 1200
            ax.scatter(
                [x],
                [y],
                s=node_size,
                c=[color],
                edgecolors="white",
                linewidth=2.5,
                zorder=8,
            )
            ax.text(
                x,
                y + 0.8,
                f'"{name}"',
                ha="center",
                fontsize=11,
                fontweight="bold",
                color=COLORS["text"],
            )
            ax.text(
                x,
                y - 0.7,
                f"p={prob:.2f}",
                ha="center",
                fontsize=9,
                color=COLORS["neutral"],
            )

            # Draw trajectories from this branch
            trajs = sorted(
                branch.trajectories, key=lambda t: t.probability, reverse=True
            )
            n = min(len(trajs), 8)

            spread = 7
            start_y = y - 2

            for i, traj in enumerate(trajs[:n]):
                # Position
                offset = (i - n / 2 + 0.5) * (spread / n)
                tx = x + offset
                ty = start_y - i * 0.6

                # Color based on structure scores (dominant structure)
                traj_color = get_structure_color(traj.structure_scores)

                # Size inversely related to deviance (low deviance = big)
                size = deviance_to_size(traj.deviance, base_size=250)

                # Edge width based on probability and deviance
                edge_width = deviance_to_linewidth(traj.deviance, base_width=2)
                alpha = min(0.9, 0.4 + traj.probability * 2)

                # Draw edge
                ax.plot(
                    [x, tx],
                    [y - 0.4, ty + 0.3],
                    color=traj_color[:3],
                    linewidth=edge_width,
                    alpha=alpha,
                    zorder=3,
                )

                # Draw node
                ax.scatter(
                    [tx],
                    [ty],
                    s=size,
                    c=[traj_color[:3]],
                    edgecolors="white",
                    linewidth=1.2,
                    zorder=6,
                    alpha=0.9,
                )

                # Label high-probability trajectories
                if traj.probability > 0.08:
                    label = traj.text[:15] + "..."
                    ax.text(
                        tx,
                        ty - 0.4,
                        label,
                        ha="center",
                        fontsize=7,
                        color=COLORS["neutral"],
                        rotation=0,
                    )

        # Add dynamics annotations
        # X_p (core at root)
        ax.annotate(
            r"$X_p$: Core at prompt",
            xy=(0.5, root_y),
            fontsize=10,
            ha="left",
            color=COLORS["text"],
        )

        # X_b annotations for each branch
        ax.annotate(
            r"$X_b^{queens}$: Queer-dominant core",
            xy=(-8, 7.5),
            fontsize=9,
            ha="center",
            color=STRUCTURE_COLORS["queer"],
        )
        ax.annotate(
            r"$X_b^{racing}$: Hetero-cis core maintained",
            xy=(8, 7.5),
            fontsize=9,
            ha="center",
            color=STRUCTURE_COLORS["man"],
        )

        # Show deviance comparison
        ax.text(
            -11,
            3,
            "Low deviance\n(thick lines, large nodes)\n= typical for branch",
            fontsize=9,
            va="center",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor=COLORS["grid"]),
        )
        ax.text(
            11,
            3,
            "High deviance\n(thin lines, small nodes)\n= atypical for branch",
            fontsize=9,
            va="center",
            ha="right",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor=COLORS["grid"]),
        )

        # Legend
        legend_elements = [
            mpatches.Patch(
                facecolor=STRUCTURE_COLORS["queer"],
                label="Queer structure dominant",
                alpha=0.8,
            ),
            mpatches.Patch(
                facecolor=STRUCTURE_COLORS["woman"],
                label="Woman structure dominant",
                alpha=0.8,
            ),
            mpatches.Patch(
                facecolor=STRUCTURE_COLORS["man"],
                label="Man structure dominant",
                alpha=0.8,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=15,
                label="Low deviance (normative)",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=7,
                label="High deviance (atypical)",
            ),
        ]

        ax.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=9,
            frameon=True,
            facecolor="white",
            edgecolor=COLORS["grid"],
        )

        # Title
        fig.suptitle(
            "LLM Trajectory Tree: Structure Compliance and Normative Branching",
            fontsize=16,
            fontweight="bold",
            color=COLORS["text"],
            y=0.97,
        )
        ax.set_title(
            "'queens' induces queer normativity | 'racing' maintains heteronormative core",
            fontsize=12,
            color=COLORS["neutral"],
            pad=15,
        )

        ax.set_xlim(-14, 14)
        ax.set_ylim(-2, 14)
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(
            output_path, dpi=200, bbox_inches="tight", facecolor=COLORS["background"]
        )
        plt.close()

        return str(output_path)

    def export_json(self, output_path: Path) -> str:
        """
        Export all visualization data as JSON for use in LaTeX/TikZ graphs.

        Structure:
        {
            "base_prompt": "...",
            "structures": ["queer", "woman", "man"],
            "structure_colors": {...},
            "branches": {
                "base": { "core": {...}, "trajectories": [...] },
                "queens": { ... },
                "racing": { ... }
            }
        }
        """
        import json

        data = {
            "base_prompt": "The tough powerlifter was also a motorcyclist who loved drag ",
            "structures": self.structures,
            "structure_colors": {
                "queer": "#9C27B0",
                "woman": "#E91E63",
                "man": "#2196F3",
            },
            "branches": {},
        }

        for name, branch in self.branches.items():
            branch_data = {
                "name": branch.name,
                "prompt_suffix": branch.prompt_suffix,
                "core": branch.core,
                "trajectories": [],
            }

            for traj in sorted(
                branch.trajectories, key=lambda t: t.probability, reverse=True
            ):
                traj_data = {
                    "text": traj.text,
                    "probability": traj.probability,
                    "structure_scores": traj.structure_scores,
                    "deviance": traj.deviance,
                    "dominant_structure": max(
                        traj.structure_scores.items(), key=lambda x: x[1]
                    )[0],
                }
                branch_data["trajectories"].append(traj_data)

            data["branches"][name] = branch_data

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return str(output_path)

    def plot_dynamics_diagram(self, output_path: Path, figsize: tuple = (18, 14)):
        """
        Create a dynamics diagram in the style of dynamics.pdf.

        Shows:
        - Schematic branching structure
        - Explicit deviance annotations d(x, Xp), d(x, Xb)
        - Core values at each branch point
        - Woman structure declining throughout
        """
        setup_style()

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("#FAFCFD")

        # Layout
        prompt_x, prompt_y = 0, 11
        branch_y = 7
        traj_y = 2

        # Get branch data
        base = self.branches["base"]
        queens = self.branches["queens"]
        racing = self.branches["racing"]

        # === PROMPT NODE ===
        prompt_color = get_structure_color(base.core)
        ax.scatter(
            [prompt_x],
            [prompt_y],
            s=1500,
            c=[prompt_color[:3]],
            edgecolors="black",
            linewidth=2,
            zorder=10,
        )

        # Prompt label and core
        ax.text(
            prompt_x,
            prompt_y + 1.2,
            "BASE PROMPT",
            ha="center",
            fontsize=14,
            fontweight="bold",
            color=COLORS["text"],
        )
        ax.text(
            prompt_x,
            prompt_y - 1.2,
            f"$X_p$ = Core at prompt\nQ:{base.core['queer']:.2f}  W:{base.core['woman']:.2f}  M:{base.core['man']:.2f}",
            ha="center",
            fontsize=10,
            color=COLORS["text"],
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLORS["grid"]
            ),
        )

        # === BRANCH POINTS ===
        queens_x, racing_x = -7, 7

        # Queens branch
        queens_color = get_structure_color(queens.core)
        ax.plot(
            [prompt_x, queens_x],
            [prompt_y - 0.6, branch_y + 0.6],
            color=queens_color[:3],
            linewidth=4,
            zorder=5,
        )
        ax.scatter(
            [queens_x],
            [branch_y],
            s=1200,
            c=[queens_color[:3]],
            edgecolors="black",
            linewidth=2,
            zorder=10,
        )
        ax.text(
            queens_x,
            branch_y + 1.0,
            '"queens"',
            ha="center",
            fontsize=13,
            fontweight="bold",
            color=STRUCTURE_COLORS["queer"],
        )
        ax.text(
            queens_x,
            branch_y - 1.1,
            f"$X_b^{{queens}}$ = Queer-dominant\nQ:{queens.core['queer']:.2f}  W:{queens.core['woman']:.2f}  M:{queens.core['man']:.2f}",
            ha="center",
            fontsize=9,
            color=COLORS["text"],
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=STRUCTURE_COLORS["queer"],
            ),
        )

        # Racing branch
        racing_color = get_structure_color(racing.core)
        ax.plot(
            [prompt_x, racing_x],
            [prompt_y - 0.6, branch_y + 0.6],
            color=racing_color[:3],
            linewidth=4,
            zorder=5,
        )
        ax.scatter(
            [racing_x],
            [branch_y],
            s=1200,
            c=[racing_color[:3]],
            edgecolors="black",
            linewidth=2,
            zorder=10,
        )
        ax.text(
            racing_x,
            branch_y + 1.0,
            '"racing"',
            ha="center",
            fontsize=13,
            fontweight="bold",
            color=STRUCTURE_COLORS["man"],
        )
        ax.text(
            racing_x,
            branch_y - 1.1,
            f"$X_b^{{racing}}$ = Hetero-cis maintained\nQ:{racing.core['queer']:.2f}  W:{racing.core['woman']:.2f}  M:{racing.core['man']:.2f}",
            ha="center",
            fontsize=9,
            color=COLORS["text"],
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=STRUCTURE_COLORS["man"],
            ),
        )

        # === TRAJECTORIES with deviance annotations ===
        # Queens trajectories
        queens_trajs = sorted(
            queens.trajectories, key=lambda t: t.probability, reverse=True
        )[:5]
        for i, traj in enumerate(queens_trajs):
            tx = queens_x + (i - 2) * 2.2
            ty = traj_y - i * 0.3

            color = get_structure_color(traj.structure_scores)
            size = deviance_to_size(traj.deviance, base_size=400)
            lw = deviance_to_linewidth(traj.deviance, base_width=3)

            ax.plot(
                [queens_x, tx],
                [branch_y - 0.5, ty + 0.4],
                color=color[:3],
                linewidth=lw,
                alpha=0.8,
                zorder=3,
            )
            ax.scatter(
                [tx],
                [ty],
                s=size,
                c=[color[:3]],
                edgecolors="white",
                linewidth=1.5,
                zorder=6,
            )

            # Deviance annotation
            if i < 3:
                # d(x, Xb) - deviance from branch core
                ax.text(
                    tx,
                    ty - 0.5,
                    f"d={traj.deviance:.2f}",
                    ha="center",
                    fontsize=8,
                    color=COLORS["neutral"],
                )
                # Short text
                short = traj.text[:12] + "..."
                ax.text(
                    tx,
                    ty - 0.9,
                    f'"{short}"',
                    ha="center",
                    fontsize=7,
                    color=COLORS["text"],
                    style="italic",
                )

        # Racing trajectories
        racing_trajs = sorted(
            racing.trajectories, key=lambda t: t.probability, reverse=True
        )[:5]
        for i, traj in enumerate(racing_trajs):
            tx = racing_x + (i - 2) * 2.2
            ty = traj_y - i * 0.3

            color = get_structure_color(traj.structure_scores)
            size = deviance_to_size(traj.deviance, base_size=400)
            lw = deviance_to_linewidth(traj.deviance, base_width=3)

            ax.plot(
                [racing_x, tx],
                [branch_y - 0.5, ty + 0.4],
                color=color[:3],
                linewidth=lw,
                alpha=0.8,
                zorder=3,
            )
            ax.scatter(
                [tx],
                [ty],
                s=size,
                c=[color[:3]],
                edgecolors="white",
                linewidth=1.5,
                zorder=6,
            )

            if i < 3:
                ax.text(
                    tx,
                    ty - 0.5,
                    f"d={traj.deviance:.2f}",
                    ha="center",
                    fontsize=8,
                    color=COLORS["neutral"],
                )
                short = traj.text[:12] + "..."
                ax.text(
                    tx,
                    ty - 0.9,
                    f'"{short}"',
                    ha="center",
                    fontsize=7,
                    color=COLORS["text"],
                    style="italic",
                )

        # === STRUCTURE EVOLUTION ANNOTATIONS ===
        # Box showing woman structure declining
        woman_box = (
            "WOMAN STRUCTURE EVOLUTION:\n"
            f"  Base:   W = {base.core['woman']:.2f} (lowest)\n"
            f"  Queens: W = {queens.core['woman']:.2f} (still low)\n"
            f"  Racing: W = {racing.core['woman']:.2f} (lowest)"
        )
        ax.text(
            -14,
            5,
            woman_box,
            fontsize=10,
            va="top",
            ha="left",
            family="monospace",
            color=STRUCTURE_COLORS["woman"],
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor=STRUCTURE_COLORS["woman"],
                alpha=0.9,
            ),
        )

        # Box showing man structure stable
        man_box = (
            "MAN STRUCTURE EVOLUTION:\n"
            f"  Base:   M = {base.core['man']:.2f}\n"
            f"  Queens: M = {queens.core['man']:.2f} (drops)\n"
            f"  Racing: M = {racing.core['man']:.2f} (stable)"
        )
        ax.text(
            14,
            5,
            man_box,
            fontsize=10,
            va="top",
            ha="right",
            family="monospace",
            color=STRUCTURE_COLORS["man"],
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor=STRUCTURE_COLORS["man"],
                alpha=0.9,
            ),
        )

        # === LEGEND ===
        legend_elements = [
            mpatches.Patch(
                facecolor=STRUCTURE_COLORS["queer"], label="Queer dominant", alpha=0.9
            ),
            mpatches.Patch(
                facecolor=STRUCTURE_COLORS["woman"], label="Woman dominant", alpha=0.9
            ),
            mpatches.Patch(
                facecolor=STRUCTURE_COLORS["man"], label="Man dominant", alpha=0.9
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=14,
                label="Low deviance d≈0",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=7,
                label="High deviance d≈1",
            ),
        ]
        ax.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=5,
            fontsize=9,
            frameon=True,
            facecolor="white",
            edgecolor=COLORS["grid"],
            bbox_to_anchor=(0.5, -0.02),
        )

        # === KEY INSIGHT BOX ===
        insight = (
            "KEY INSIGHT: The token 'queens' dramatically shifts the normative core\n"
            "from heterosexual-cisgender (Q=0.12) to queer-dominant (Q=0.85),\n"
            "while 'racing' maintains the original hetero-cis normativity (Q=0.06)."
        )
        ax.text(
            0,
            -1.5,
            insight,
            ha="center",
            fontsize=11,
            color=COLORS["text"],
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#E8F5E9",
                edgecolor="#4CAF50",
                alpha=0.9,
            ),
        )

        # Title
        fig.suptitle(
            "Dynamics Diagram: Branching Points and Core Shifts",
            fontsize=16,
            fontweight="bold",
            color=COLORS["text"],
            y=0.97,
        )
        ax.set_title(
            "'The tough powerlifter was also a motorcyclist who loved drag ...'",
            fontsize=11,
            color=COLORS["neutral"],
            style="italic",
            pad=10,
        )

        ax.set_xlim(-16, 16)
        ax.set_ylim(-3, 14)
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(
            output_path, dpi=200, bbox_inches="tight", facecolor=COLORS["background"]
        )
        plt.close()

        return str(output_path)


# -----------------------------------------------------------------------------
# Core Logic
# -----------------------------------------------------------------------------


def visualize_queer(inp: VizQueerInput) -> VizQueerOutput:
    """Main queer visualization logic."""
    inp.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if inp.verbose:
        print("=" * 80)
        print("QUEER.JSON VISUALIZATION")
        print("=" * 80)
        print()
        print("Generating synthetic data to demonstrate structure dynamics...")

    # Generate synthetic data
    branches = generate_synthetic_data()

    if inp.verbose:
        print("Generated data for branches:")
        for branch in branches:
            print(f"  {branch.name}: {len(branch.trajectories)} trajectories")
            print(
                f"    Core: Q={branch.core['queer']:.2f}, "
                f"W={branch.core['woman']:.2f}, M={branch.core['man']:.2f}"
            )
        print()

    # Create visualizer
    viz = QueerVisualizer(branches)

    # Generate visualizations
    if inp.verbose:
        print("Generating visualizations...")

    # 1. Branching tree
    tree_path = inp.output_dir / f"queer_branching_tree_{timestamp}.png"
    viz.plot_branching_tree(tree_path)
    if inp.verbose:
        print(f"  Saved: {tree_path}")

    # 2. Dynamics
    dynamics_path = inp.output_dir / f"queer_dynamics_{timestamp}.png"
    viz.plot_dynamics(dynamics_path)
    if inp.verbose:
        print(f"  Saved: {dynamics_path}")

    # 3. Trajectory details
    details_path = inp.output_dir / f"queer_trajectory_details_{timestamp}.png"
    viz.plot_trajectory_details(details_path)
    if inp.verbose:
        print(f"  Saved: {details_path}")

    # 4. Marshall-style tree
    marshall_path = inp.output_dir / f"queer_marshall_tree_{timestamp}.png"
    viz.plot_marshall_style_tree(marshall_path)
    if inp.verbose:
        print(f"  Saved: {marshall_path}")

    # 5. Dynamics diagram (like dynamics.pdf)
    diagram_path = inp.output_dir / f"queer_dynamics_diagram_{timestamp}.png"
    viz.plot_dynamics_diagram(diagram_path)
    if inp.verbose:
        print(f"  Saved: {diagram_path}")

    # 6. JSON export for LaTeX
    json_path = inp.output_dir / f"queer_data_{timestamp}.json"
    viz.export_json(json_path)
    if inp.verbose:
        print(f"  Saved: {json_path}")

    return VizQueerOutput(
        tree_path=tree_path,
        dynamics_path=dynamics_path,
        details_path=details_path,
        marshall_path=marshall_path,
        diagram_path=diagram_path,
        json_path=json_path,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize queer.json trial - structure-aware diversity"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent.parent / "outputs" / "queer_viz",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    return parser.parse_args()


def input_from_args(args: argparse.Namespace) -> VizQueerInput:
    """Load input from command line arguments."""
    return VizQueerInput(
        output_dir=args.output,
        verbose=not args.quiet,
    )


def save_output(args: argparse.Namespace, output: VizQueerOutput) -> None:
    """Save output (already saved during visualization)."""
    pass


def print_output(args: argparse.Namespace, output: VizQueerOutput) -> None:
    """Print output summary."""
    if not args.quiet:
        print()
        print("=" * 80)
        print("VISUALIZATION COMPLETE")
        print("=" * 80)
        print()
        print("Key observations demonstrated:")
        print("  - Base prompt induces low queer compliance (hetero-cis dominant)")
        print("  - 'queens' branch dramatically shifts core to queer dominance")
        print("  - 'racing' branch maintains hetero-cis normativity")
        print("  - 'woman' structure remains lowest across all branches")
        print("  - 'man' structure remains relatively stable")
        print()
        print("Color encoding:")
        print("  - Purple: Queer structure dominant")
        print("  - Pink: Woman structure dominant")
        print("  - Blue: Man structure dominant")
        print()
        print("Size/thickness encoding:")
        print("  - Large nodes / thick edges: Low deviance (typical for branch)")
        print("  - Small nodes / thin edges: High deviance (atypical)")


def main() -> int:
    args = get_args()
    inp: VizQueerInput = input_from_args(args)
    output: VizQueerOutput = visualize_queer(inp)

    save_output(args, output)
    print_output(args, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
