"""
Brute-force searcher for trajectory exploration.

Runs an explorer (greedy or sampling) multiple times to collect trajectories
until a target probability mass is reached. Reuses a shared tree for efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from xenotechnics.common import String
from xenotechnics.trees.tree import TreeNode

from ..common import AbstractGenerator


@dataclass
class SearchResult:
    """Result from brute-force search."""

    trajectories: List[TreeNode]
    probabilities: np.ndarray
    total_mass: float
    tree: TreeNode


class BruteSearcher:
    """
    Brute-force trajectory searcher.

    Runs a generator multiple times to collect trajectories until
    the cumulative probability mass reaches a threshold.

    Uses a shared tree across runs for efficiency - the generator
    reuses the existing tree structure instead of rebuilding.

    Args:
        generator: Generator instance (GreedyGenerator or SamplingGenerator)
    """

    def __init__(self, generator: AbstractGenerator):
        self.generator = generator

    def search(
        self,
        prompt: Optional[String] = None,
        min_probability_mass: float = 0.9,
        max_new_tokens: Optional[int] = 100,
        max_trajectories: Optional[int] = None,
        verbose: bool = True,
        **generator_kwargs,
    ) -> SearchResult:
        """
        Search by collecting trajectories until mass threshold.

        Args:
            prompt: Starting prompt
            min_probability_mass: Stop when this mass is reached
            max_new_tokens: Maximum tokens per trajectory (None = no limit)
            max_trajectories: Maximum trajectories to collect (None = no limit)
            verbose: Print progress
            **generator_kwargs: Additional args for generator (e.g., temperature, seed)

        Returns:
            SearchResult with trajectories and probabilities
        """
        if verbose:
            print()
            print("=" * 70)
            print("  BRUTE FORCE SEARCH")
            print("=" * 70)
            print()
            print(f"  Target mass:       {min_probability_mass}")
            print(
                f"  Max new tokens:    {max_new_tokens if max_new_tokens else 'unlimited'}"
            )
            print(
                f"  Max trajectories:  {max_trajectories if max_trajectories else 'unlimited'}"
            )
            print()
            print("-" * 70)
            print(f"  {'#':<4}  {'Prob':>12}  {'Total Mass':>12}  {'Continuation':<30}")
            print("-" * 70)

        # Get prompt text length for display
        prompt_text = prompt.to_text() if prompt else ""
        prompt_len = len(prompt_text)

        # First run builds the tree
        base_seed = generator_kwargs.get("seed", 42)

        # Handle None max_new_tokens
        run_max_tokens = max_new_tokens if max_new_tokens is not None else 9999999999

        tree = self.generator.run(
            prompt=prompt,
            max_new_tokens=run_max_tokens,
            verbose=False,
            existing_tree=None,
            **generator_kwargs,
        )

        # Track progress using tree's trajectory nodes
        iteration = 1
        prev_count = 0

        while True:
            # Get all trajectories from the shared tree
            trajectory_nodes = tree.get_trajectory_nodes()
            total_mass = sum(
                t.get_continuation_prob(self.generator.prompt_token_count)
                for t in trajectory_nodes
            )

            # Print new trajectories
            if verbose and len(trajectory_nodes) > prev_count:
                for i, traj_node in enumerate(
                    trajectory_nodes[prev_count:], prev_count + 1
                ):
                    cont_prob = traj_node.get_continuation_prob(
                        self.generator.prompt_token_count
                    )
                    # Get only the continuation part (after prompt)
                    full_text = traj_node.string.to_text()
                    continuation = full_text[prompt_len:].strip()
                    # Truncate for display
                    display_text = (
                        continuation[-40:] if len(continuation) > 40 else continuation
                    )
                    display_text = display_text.replace("\n", "\\n")
                    print(
                        f"  {i:<4}  {cont_prob:>12.4e}  {total_mass:>12.4f}  {display_text}"
                    )
                prev_count = len(trajectory_nodes)

            # Check stopping conditions
            if total_mass >= min_probability_mass:
                break
            if (
                max_trajectories is not None
                and len(trajectory_nodes) >= max_trajectories
            ):
                break

            # Run explorer again with shared tree
            iteration += 1
            if "seed" in generator_kwargs or base_seed:
                generator_kwargs["seed"] = base_seed + iteration

            self.generator.run(
                prompt=prompt,
                max_new_tokens=run_max_tokens,
                verbose=False,
                existing_tree=tree,
                **generator_kwargs,
            )

        # Final trajectory list and probabilities
        trajectory_nodes = tree.get_trajectory_nodes()
        probabilities = np.array(
            [
                t.get_continuation_prob(self.generator.prompt_token_count)
                for t in trajectory_nodes
            ]
        )
        total_mass = float(probabilities.sum())

        if verbose:
            print("-" * 70)
            print()
            print(f"  Trajectories collected:  {len(trajectory_nodes)}")
            print(f"  Total probability mass:  {total_mass:.4f}")
            print()
            print("=" * 70)
            print()

        return SearchResult(
            trajectories=trajectory_nodes,
            probabilities=probabilities,
            total_mass=total_mass,
            tree=tree,
        )
