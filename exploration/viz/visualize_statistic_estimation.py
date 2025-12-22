"""
Visualization tools for StatisticsEstimation results.

Provides functions to visualize:
- LLM generation trees (Section 3.1)
- System cores (Section 3.3, Eq. 5-6)
- Orientations and deviances (Section 3.3, Eq. 7-8)
- Homogenization metrics (Section 4)
- Dynamics of diversity (Section 3.5)

Paper reference: "Structure-aware Diversity Pursuit as AI Safety strategy
against Homogenization"
"""

import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

# Try importing networkx for tree visualization
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed. Tree visualization will be limited.")


def load_results(json_path: str) -> Dict[str, Any]:
    """
    Load StatisticsEstimation results from JSON file.

    Args:
        json_path: Path to JSON results file

    Returns:
        Dictionary with results
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def visualize_tree_structure(
    results: Dict[str, Any],
    output_path: Optional[str] = None,
    max_nodes: int = 100
) -> Figure:
    """
    Visualize LLM generation tree structure.

    Paper (Section 3.1): "Any LLM induces a tree on Str: the root is ⊥,
    each node is a string, the leaves are trajectories."

    Args:
        results: Results dictionary
        output_path: Optional path to save figure
        max_nodes: Maximum nodes to display (for readability)

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    tree_info = results.get('tree_info', {})
    num_nodes = tree_info.get('num_nodes', 0)
    max_depth = tree_info.get('max_depth', 0)
    num_trajectories = tree_info.get('num_trajectories', 0)

    # Display tree statistics
    stats_text = f"""
    LLM Generation Tree Statistics

    Total Nodes: {num_nodes}
    Maximum Depth: {max_depth}
    Complete Trajectories: {num_trajectories}
    Total Probability Mass: {tree_info.get('total_mass', 0):.4f}
    """

    ax.text(0.5, 0.5, stats_text, transform=ax.transAxes,
            fontsize=14, verticalalignment='center',
            horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.axis('off')
    ax.set_title("LLM Generation Tree Overview", fontsize=16, fontweight='bold')

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def visualize_cores(
    results: Dict[str, Any],
    output_path: Optional[str] = None
) -> Figure:
    """
    Visualize system cores across different systems.

    Paper (Section 3.3, Eq. 6): "System core: ⟨Λ_n⟩ = Σ p(y)Λ_n(y)"

    "If the system core tells us what is normatively complied with,
    orientations tell us in what ways a string is non-normative."

    Args:
        results: Results dictionary
        output_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    core_infos = results.get('core_infos', [])
    system_infos = results.get('system_infos', [])

    if not core_infos:
        print("No core information found in results")
        return None

    num_systems = len(core_infos)
    fig, axes = plt.subplots(1, num_systems, figsize=(6 * num_systems, 5))
    if num_systems == 1:
        axes = [axes]

    fig.suptitle("System Cores: Expected Structural Compliance ⟨Λ_n⟩",
                 fontsize=16, fontweight='bold')

    for idx, (core_info, system_info) in enumerate(zip(core_infos, system_infos)):
        ax = axes[idx]
        core_vector = np.array(core_info['core_vector'])

        # Get structure names if available
        structure_names = system_info.get('structure_names')
        if structure_names:
            x_labels = [name[:30] + '...' if len(name) > 30 else name
                       for name in structure_names]
        else:
            x_labels = [f"α_{i+1}" for i in range(len(core_vector))]

        # Bar plot of cores
        bars = ax.bar(range(len(core_vector)), core_vector, alpha=0.7, color='steelblue')
        ax.set_xlabel("Structures", fontsize=12)
        ax.set_ylabel("Core Value ⟨α_i⟩", fontsize=12)
        ax.set_title(f"{system_info.get('type', 'System')} (n={len(core_vector)})",
                    fontsize=12)
        ax.set_xticks(range(len(core_vector)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_ylim([0, 1.0])
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Midpoint')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def visualize_deviances(
    results: Dict[str, Any],
    output_path: Optional[str] = None
) -> Figure:
    """
    Visualize deviance distributions across systems.

    Paper (Section 3.3, Eq. 8): "Deviance: ∥θ_n(x)∥_θ = ∂_n(x)"
    Paper (Section 4): "Homogenization as minimizing deviance"

    Args:
        results: Results dictionary
        output_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    deviance_infos = results.get('deviance_infos', [])
    system_infos = results.get('system_infos', [])

    if not deviance_infos:
        print("No deviance information found in results")
        return None

    num_systems = len(deviance_infos)
    fig, axes = plt.subplots(2, num_systems, figsize=(6 * num_systems, 10))
    if num_systems == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle("Deviance Analysis: Non-Normativity Measures ∂_n(x)",
                 fontsize=16, fontweight='bold')

    for idx, (dev_info, system_info) in enumerate(zip(deviance_infos, system_infos)):
        # Histogram
        ax_hist = axes[0, idx]
        deviances = np.array(dev_info['deviances'])

        ax_hist.hist(deviances, bins=30, alpha=0.7, color='coral', edgecolor='black')
        ax_hist.axvline(dev_info['expected_deviance'], color='red', linestyle='--',
                       linewidth=2, label=f"E[∂] = {dev_info['expected_deviance']:.4f}")
        ax_hist.axvline(dev_info['expected_deviance'] + dev_info['std_deviance'],
                       color='orange', linestyle=':', alpha=0.7,
                       label=f"±σ = {dev_info['std_deviance']:.4f}")
        ax_hist.axvline(dev_info['expected_deviance'] - dev_info['std_deviance'],
                       color='orange', linestyle=':', alpha=0.7)

        ax_hist.set_xlabel("Deviance ∂_n", fontsize=11)
        ax_hist.set_ylabel("Frequency", fontsize=11)
        ax_hist.set_title(f"{system_info.get('type', 'System')}: Distribution",
                         fontsize=12)
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)

        # Box plot with statistics
        ax_box = axes[1, idx]
        ax_box.boxplot([deviances], labels=['Deviances'], vert=True)

        stats_text = f"""
        Expected: {dev_info['expected_deviance']:.4f}
        Variance: {dev_info['variance_deviance']:.4f}
        Std Dev: {dev_info['std_deviance']:.4f}
        Min: {dev_info['min_deviance']:.4f}
        Max: {dev_info['max_deviance']:.4f}
        """
        ax_box.text(1.5, 0.5, stats_text, transform=ax_box.transAxes,
                   fontsize=9, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        ax_box.set_ylabel("Deviance ∂_n", fontsize=11)
        ax_box.set_title(f"{system_info.get('type', 'System')}: Statistics",
                        fontsize=12)
        ax_box.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def visualize_core_comparison(
    results_list: List[Dict[str, Any]],
    labels: List[str],
    output_path: Optional[str] = None
) -> Figure:
    """
    Compare cores across multiple experiments/prompts.

    Paper (Section 3.4): "Different prompts collapse to different modes."
    "Different prompts may differ substantially."

    Args:
        results_list: List of results dictionaries
        labels: Labels for each result
        output_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    if not results_list:
        print("No results provided for comparison")
        return None

    # Get number of systems from first result
    num_systems = len(results_list[0].get('core_infos', []))

    fig, axes = plt.subplots(1, num_systems, figsize=(8 * num_systems, 6))
    if num_systems == 1:
        axes = [axes]

    fig.suptitle("Core Comparison Across Prompts: ⟨Λ_n⟩(x_p)",
                 fontsize=16, fontweight='bold')

    for sys_idx in range(num_systems):
        ax = axes[sys_idx]

        # Extract cores for this system from all results
        all_cores = []
        for results in results_list:
            core_info = results['core_infos'][sys_idx]
            all_cores.append(np.array(core_info['core_vector']))

        # Get system info
        system_info = results_list[0]['system_infos'][sys_idx]
        structure_names = system_info.get('structure_names')

        # Plot grouped bars
        x = np.arange(len(all_cores[0]))
        width = 0.8 / len(results_list)

        for i, (core, label) in enumerate(zip(all_cores, labels)):
            offset = (i - len(results_list)/2) * width + width/2
            ax.bar(x + offset, core, width, label=label, alpha=0.7)

        # Formatting
        if structure_names:
            x_labels = [name[:20] + '...' if len(name) > 20 else name
                       for name in structure_names]
        else:
            x_labels = [f"α_{i+1}" for i in range(len(x))]

        ax.set_xlabel("Structures", fontsize=12)
        ax.set_ylabel("Core Value ⟨α_i⟩", fontsize=12)
        ax.set_title(f"{system_info.get('type', 'System')}",
                    fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_ylim([0, 1.0])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def visualize_homogenization_metrics(
    results: Dict[str, Any],
    output_path: Optional[str] = None
) -> Figure:
    """
    Visualize homogenization metrics.

    Paper (Section 4, Eq. 12-13): "Homogenization as minimizing all deviance"
    E[∂] → 0 and Var[∂] → 0

    Args:
        results: Results dictionary
        output_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    deviance_infos = results.get('deviance_infos', [])
    system_infos = results.get('system_infos', [])

    if not deviance_infos:
        print("No deviance information found")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Homogenization Metrics",
                 fontsize=16, fontweight='bold')

    system_labels = [info.get('type', f'System {i}')
                    for i, info in enumerate(system_infos)]

    # Expected deviance
    expected_devs = [info['expected_deviance'] for info in deviance_infos]
    ax1.bar(range(len(expected_devs)), expected_devs, alpha=0.7, color='indianred')
    ax1.set_xlabel("System", fontsize=12)
    ax1.set_ylabel("E[∂_n]", fontsize=12)
    ax1.set_title("Expected Deviance\n(Homogenization → 0)", fontsize=13)
    ax1.set_xticks(range(len(expected_devs)))
    ax1.set_xticklabels(system_labels, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')

    # Deviance variance
    var_devs = [info['variance_deviance'] for info in deviance_infos]
    ax2.bar(range(len(var_devs)), var_devs, alpha=0.7, color='darkseagreen')
    ax2.set_xlabel("System", fontsize=12)
    ax2.set_ylabel("Var[∂_n]", fontsize=12)
    ax2.set_title("Deviance Variance\n(Homogenization → 0)", fontsize=13)
    ax2.set_xticks(range(len(var_devs)))
    ax2.set_xticklabels(system_labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def create_full_report(
    json_path: str,
    output_dir: Optional[str] = None
):
    """
    Create a full visualization report from a JSON results file.

    Generates all visualizations and saves them to output directory.

    Args:
        json_path: Path to JSON results file
        output_dir: Directory to save visualizations (default: same as JSON)
    """
    # Load results
    results = load_results(json_path)

    # Determine output directory
    if output_dir is None:
        output_dir = Path(json_path).parent / "visualizations"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Base name for outputs
    base_name = Path(json_path).stem

    print(f"Creating visualizations for: {json_path}")
    print(f"Output directory: {output_dir}")

    # Generate all visualizations
    print("\n1. Visualizing tree structure...")
    fig_tree = visualize_tree_structure(
        results,
        output_path=str(output_dir / f"{base_name}_tree.png")
    )
    plt.close(fig_tree)

    print("2. Visualizing cores...")
    fig_cores = visualize_cores(
        results,
        output_path=str(output_dir / f"{base_name}_cores.png")
    )
    if fig_cores:
        plt.close(fig_cores)

    print("3. Visualizing deviances...")
    fig_deviances = visualize_deviances(
        results,
        output_path=str(output_dir / f"{base_name}_deviances.png")
    )
    if fig_deviances:
        plt.close(fig_deviances)

    print("4. Visualizing homogenization metrics...")
    fig_metrics = visualize_homogenization_metrics(
        results,
        output_path=str(output_dir / f"{base_name}_homogenization.png")
    )
    if fig_metrics:
        plt.close(fig_metrics)

    print(f"\n✓ All visualizations saved to: {output_dir}")


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Visualize StatisticsEstimation results"
    )
    parser.add_argument(
        "json_path",
        type=str,
        help="Path to JSON results file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for visualizations (default: same as JSON)"
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs="+",
        help="Additional JSON files to compare"
    )

    args = parser.parse_args()

    if args.compare:
        # Comparison mode
        all_paths = [args.json_path] + args.compare
        results_list = [load_results(p) for p in all_paths]
        labels = [Path(p).stem for p in all_paths]

        output_dir = Path(args.output_dir) if args.output_dir else Path("./visualizations")
        output_dir.mkdir(exist_ok=True, parents=True)

        print("Creating comparison visualization...")
        fig = visualize_core_comparison(
            results_list,
            labels,
            output_path=str(output_dir / "core_comparison.png")
        )
        if fig:
            plt.close(fig)
        print(f"✓ Comparison saved to: {output_dir / 'core_comparison.png'}")

    else:
        # Single file mode
        create_full_report(args.json_path, args.output_dir)


if __name__ == "__main__":
    main()
