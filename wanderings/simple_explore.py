"""
Simple exploration script using Explorer abstraction.

Runs greedy generation and builds TreeNode.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from exploration import GreedyExplorer
from xenotechnics.common import String


def main():
    # Define prompt
    prompt_text = "Complete the following sentence in less than 3 words. Just give me completion: Roses are"
    prompt = String.from_text(prompt_text)

    # Create greedy explorer
    model_name = "google/gemma-2-2b-it"

    print("=" * 60)
    print("GREEDY EXPLORATION")
    print("=" * 60)
    print()

    greedy_explorer = GreedyExplorer(model_name=model_name)
    greedy_tree = greedy_explorer.run(prompt=prompt, max_new_tokens=50, verbose=True)

    # Analyze greedy tree
    print("\n" + "=" * 60)
    print("Greedy TreeNode Analysis:")
    print("=" * 60)
    print(f"Root depth: {greedy_tree.depth()}")

    trajectory_nodes = greedy_tree.get_trajectory_nodes()
    print(f"Number of trajectories: {len(trajectory_nodes)}")

    for i, trajectory_node in enumerate(trajectory_nodes):
        print("\n\nTrajectory" + str(i) + "\n")
        print(trajectory_node)

        # Print continuation probability (from prompt to end)
        cont_logprob = trajectory_node.get_continuation_logprob(
            greedy_explorer.prompt_token_count
        )
        cont_prob = trajectory_node.get_continuation_prob(
            greedy_explorer.prompt_token_count
        )
        print(
            f"\n  Continuation (from prompt): prob={cont_prob:.6e} | logprob={cont_logprob:.4f}"
        )

    return


if __name__ == "__main__":
    main()
