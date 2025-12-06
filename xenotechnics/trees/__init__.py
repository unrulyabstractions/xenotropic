"""
Tree structures for LLM generation.

Key concepts:
- LLMTree: Singleton per LLM (use get_tree(llm_id))
- TreeNode: Represents a prefix in the generation tree
- Subtrees are just TreeNode references (shared structure)
- Full next-token distribution stored in each node
- Log probabilities for numerical stability
- next() method samples from distribution with various strategies

Usage:
    # Get or create tree for an LLM
    tree = LLMTree.get_tree("gpt-4", tokenizer=tokenizer)

    # Set distribution at current node (from LLM)
    tree.root.set_distribution(logits=model_logits)

    # Sample next token (creates child node)
    node = tree.root.next(tokenizer, greedy=True)

    # Or sample with temperature
    node = tree.root.next(tokenizer, temperature=0.7, top_k=50)

    # Greedy path traversal
    path = tree.root.greedy_path(tokenizer, max_depth=100)

    # Get subtree (just a node reference)
    subtree = tree.get_node(prefix_string)
"""

from .tree import TreeNode, LLMTree

__all__ = ['TreeNode', 'LLMTree']
