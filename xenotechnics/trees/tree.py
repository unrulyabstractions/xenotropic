"""
Tree implementation for LLM generation.

Key design principles:
- LLMTree is a singleton per LLM (stores only root node)
- TreeNode stores full next-token distribution (not just children)
- next() method samples from distribution with various strategies
- Log probabilities for numerical stability
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from xenotechnics.common import AbstractSystem, String


@dataclass
class TreeNode:
    """
    Node in an LLM generation tree.

    Design:
    - Each node represents a string (prefix)
    - Stores FULL next-token distribution from LLM
    - Children created dynamically as we traverse
    - Edge probabilities stored as log probs

    Attributes:
        string: String at this node
        parent: Parent node (None for root)
        children: Dict mapping token -> child node
        child_logprobs: Dict mapping token -> log p(token|parent)
        next_token_logits: Full next-token logits over vocabulary
        next_token_distribution: Full next-token probabilities (softmax of logits)
        metadata: Optional metadata storage
    """

    string: String
    parent: Optional[TreeNode] = None
    children: Dict[str, TreeNode] = field(default_factory=dict)
    child_logprobs: Dict[str, float] = field(default_factory=dict)
    next_token_logits: Optional[np.ndarray] = None
    next_token_distribution: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _is_trajectory: bool = False

    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0

    def is_trajectory(self) -> bool:
        """Check if this node represents a complete trajectory."""
        return self._is_trajectory

    def mark_as_trajectory(self) -> None:
        """Mark this node as a complete trajectory."""
        self._is_trajectory = True

    def depth(self) -> int:
        """Compute depth by traversing to root."""
        if self.parent is None:
            return 0
        return 1 + self.parent.depth()

    def set_distribution(
        self,
        logits: np.ndarray = None,
        probs: np.ndarray = None,
        temperature: float = 1.0,
    ):
        """
        Set the next-token distribution for this node.

        Args:
            logits: Raw logits from LLM (will be softmaxed)
            probs: Pre-computed probabilities (if logits not provided)
            temperature: Temperature for softmax (only used if logits provided)
        """
        if logits is not None:
            self.next_token_logits = np.array(logits)
            # Apply temperature and compute probabilities
            scaled_logits = logits / temperature
            # Numerical stability: subtract max
            scaled_logits = scaled_logits - np.max(scaled_logits)
            exp_logits = np.exp(scaled_logits)
            self.next_token_distribution = exp_logits / np.sum(exp_logits)
        elif probs is not None:
            self.next_token_distribution = np.array(probs)
            # Compute logits from probs (inverse operation)
            self.next_token_logits = np.log(probs + 1e-10)
        else:
            raise ValueError("Must provide either logits or probs")

    def next(
        self,
        tokenizer,
        greedy: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        token_id: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> TreeNode:
        """
        Sample next token and return/create child node.

        This is the main traversal method. Supports multiple sampling strategies:
        - greedy: Take argmax
        - temperature: Sample with temperature scaling
        - top_k: Sample from top-k tokens
        - top_p: Nucleus sampling (sample from top-p cumulative probability)
        - token_id: Force specific token (for deterministic traversal)

        Args:
            tokenizer: Tokenizer for decoding token IDs
            greedy: If True, take argmax (deterministic)
            temperature: Temperature for sampling (higher = more random)
            top_k: Only sample from top k tokens
            top_p: Nucleus sampling threshold
            token_id: Force this specific token ID
            seed: Random seed for reproducibility

        Returns:
            Child TreeNode (created if doesn't exist)
        """
        if self.next_token_distribution is None:
            raise ValueError(
                "No distribution set for this node. Call set_distribution() first."
            )

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Get probability distribution
        probs = self.next_token_distribution.copy()

        # Apply temperature scaling if not greedy
        if not greedy and temperature != 1.0:
            # Recompute probs with temperature
            logits = self.next_token_logits / temperature
            logits = logits - np.max(logits)
            probs = np.exp(logits) / np.sum(np.exp(logits))

        # Select token based on strategy
        if token_id is not None:
            # Force specific token
            sampled_token_id = token_id
        elif greedy:
            # Greedy: take argmax
            sampled_token_id = int(np.argmax(probs))
        else:
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                top_k_indices = np.argsort(probs)[-top_k:]
                filtered_probs = np.zeros_like(probs)
                filtered_probs[top_k_indices] = probs[top_k_indices]
                probs = filtered_probs / np.sum(filtered_probs)

            # Apply top-p (nucleus) filtering
            if top_p is not None and 0 < top_p < 1:
                sorted_indices = np.argsort(probs)[::-1]
                sorted_probs = probs[sorted_indices]
                cumsum_probs = np.cumsum(sorted_probs)
                cutoff_index = np.searchsorted(cumsum_probs, top_p) + 1
                top_p_indices = sorted_indices[:cutoff_index]
                filtered_probs = np.zeros_like(probs)
                filtered_probs[top_p_indices] = probs[top_p_indices]
                probs = filtered_probs / np.sum(filtered_probs)

            # Sample from distribution
            sampled_token_id = int(np.random.choice(len(probs), p=probs))

        # Decode token
        token = tokenizer.decode([sampled_token_id])

        # Get log probability of sampled token
        logprob = float(np.log(probs[sampled_token_id] + 1e-10))

        # Check if child already exists
        if token in self.children:
            # Update log probability
            self.child_logprobs[token] = logprob
            return self.children[token]

        # Create new child
        new_string = self.string.extend_with_token_id(token, sampled_token_id)
        child = TreeNode(
            string=new_string, parent=self, metadata={"token_id": sampled_token_id}
        )

        # Store child and its log probability
        self.children[token] = child
        self.child_logprobs[token] = logprob

        return child

    def add_child(
        self,
        token: str,
        logprob: float,
        token_id: Optional[int] = None,
        metadata: Dict[str, Any] = None,
    ) -> TreeNode:
        """
        Manually add a child node (for building tree without sampling).

        Args:
            token: Token string
            logprob: Log probability of this token given parent
            token_id: Token ID (optional)
            metadata: Optional metadata for the child

        Returns:
            The new child node (or existing if token already present)
        """
        # Check if child already exists
        if token in self.children:
            # Update log probability and return existing child
            self.child_logprobs[token] = logprob
            if metadata:
                self.children[token].metadata.update(metadata)
            return self.children[token]

        # Create new child
        new_string = self.string.extend(token)
        child_metadata = {"token_id": token_id} if token_id else {}
        if metadata:
            child_metadata.update(metadata)

        child = TreeNode(string=new_string, parent=self, metadata=child_metadata)

        # Store child and its log probability
        self.children[token] = child
        self.child_logprobs[token] = logprob

        return child

    def set_child_logprobs_from_distribution(
        self, distribution: np.ndarray, tokenizer, min_logprob: float = -20.0
    ) -> None:
        """
        Set child_logprobs from a full distribution.

        Converts probability distribution over vocabulary to log probabilities
        and stores them in child_logprobs dict. Does NOT overwrite existing
        logprobs to preserve values from previous sampling runs.

        Args:
            distribution: Probability distribution over vocabulary
            tokenizer: Tokenizer for decoding token IDs
            min_logprob: Minimum log probability to store (for efficiency)
        """
        # Don't clear - preserve existing logprobs from previous runs
        for token_id, prob in enumerate(distribution):
            if prob > 0:
                logprob = float(np.log(prob))
                if logprob >= min_logprob:
                    token_str = tokenizer.decode([token_id])
                    # Only set if not already present
                    if token_str not in self.child_logprobs:
                        self.child_logprobs[token_str] = logprob

    def get_child(self, token: str) -> Optional[TreeNode]:
        """Get child node for a given token."""
        return self.children.get(token)

    def path_to_root(self) -> List[TreeNode]:
        """Get path from this node to root."""
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))

    def path_logprob(self) -> float:
        """
        Compute log probability of path from root to this node.

        Returns:
            Sum of log probabilities along path from root
        """
        logprob_sum = 0.0
        current = self

        while current.parent is not None:
            # Get the token that led to this node
            token = current.string.tokens[-1]
            # Add the log probability from parent to this node
            logprob_sum += current.parent.child_logprobs.get(token, -np.inf)
            current = current.parent

        return logprob_sum

    def probability(self) -> float:
        """
        Compute probability of path from root to this node.

        Returns:
            exp(path_logprob())
        """
        return np.exp(self.path_logprob())

    def greedy_path(self, tokenizer, max_depth: int = 100) -> List[TreeNode]:
        """
        Find greedy path from this node.

        Convenience method that calls next(greedy=True) repeatedly.

        Args:
            tokenizer: Tokenizer for decoding
            max_depth: Maximum depth to traverse

        Returns:
            List of nodes in greedy path starting from this node
        """
        path = [self]
        current = self

        for _ in range(max_depth):
            if current.is_trajectory():
                break

            if current.next_token_distribution is None:
                break

            # Call next with greedy=True
            try:
                current = current.next(tokenizer, greedy=True)
                path.append(current)
            except Exception:
                break

        return path

    def sample_path(
        self,
        tokenizer,
        max_depth: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> List[TreeNode]:
        """
        Sample a path from this node.

        Args:
            tokenizer: Tokenizer for decoding
            max_depth: Maximum depth to traverse
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            seed: Random seed

        Returns:
            List of nodes in sampled path
        """
        path = [self]
        current = self

        for _ in range(max_depth):
            if current.is_trajectory():
                break

            if current.next_token_distribution is None:
                break

            try:
                current = current.next(
                    tokenizer,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    seed=seed,
                )
                path.append(current)
            except Exception:
                break

        return path

    def branch_mass_logprob(self) -> float:
        """
        Compute log of total probability mass in this subtree.

        Uses log-sum-exp trick for numerical stability.

        Returns:
            Log of total probability mass of all leaf descendants
        """
        if self.is_leaf():
            return self.path_logprob()

        # Collect log probs of all leaf descendants
        leaf_logprobs = []

        def collect_leaves(node: TreeNode):
            if node.is_leaf():
                leaf_logprobs.append(node.path_logprob())
            else:
                for child in node.children.values():
                    collect_leaves(child)

        collect_leaves(self)

        if not leaf_logprobs:
            return -np.inf

        # Log-sum-exp for numerical stability
        max_logprob = max(leaf_logprobs)
        if max_logprob == -np.inf:
            return -np.inf

        sum_exp = sum(np.exp(lp - max_logprob) for lp in leaf_logprobs)
        return max_logprob + np.log(sum_exp)

    def branch_mass(self) -> float:
        """Total probability mass of all descendant leaf nodes."""
        return np.exp(self.branch_mass_logprob())

    def get_all_descendants(self) -> List[TreeNode]:
        """Get all descendant nodes via DFS."""
        descendants = []

        def traverse(node: TreeNode):
            for child in node.children.values():
                descendants.append(child)
                traverse(child)

        traverse(self)
        return descendants

    def get_trajectories(self) -> List[String]:
        """Get all complete trajectory strings in this subtree."""
        trajectories = []

        def traverse(node: TreeNode):
            if node.is_trajectory():
                trajectories.append(node.string)
            for child in node.children.values():
                traverse(child)

        traverse(self)
        return trajectories

    def get_trajectory_nodes(self) -> List[TreeNode]:
        """Get all trajectory nodes in this subtree."""
        trajectory_nodes = []

        def traverse(node: TreeNode):
            if node.is_trajectory():
                trajectory_nodes.append(node)
            for child in node.children.values():
                traverse(child)

        traverse(self)
        return trajectory_nodes

    def get_conditional_logprob(self, prompt: String) -> float:
        """
        Get log P(this trajectory | prompt).

        Computes the sum of log probabilities for tokens generated
        after the prompt.

        Args:
            prompt: The prompt string

        Returns:
            Log probability of continuation after prompt
        """
        return self.get_continuation_logprob(len(prompt.tokens))

    def get_continuation_logprob(self, prompt_token_count: int) -> float:
        """
        Get log P(continuation | prompt) given prompt token count.

        Computes the sum of log probabilities for tokens generated
        after the prompt.

        Args:
            prompt_token_count: Number of prompt tokens (edges from root)

        Returns:
            Log probability of continuation after prompt
        """

        # Traverse from this node back to root, collecting log probs
        path_logprobs = []
        current = self
        while current.parent is not None:
            token = current.string.tokens[-1]
            path_logprobs.append(current.parent.child_logprobs.get(token, -np.inf))
            current = current.parent

        # Reverse to get root → trajectory order
        path_logprobs.reverse()

        # Sum log probs for tokens after the prompt
        if len(path_logprobs) > prompt_token_count:
            return sum(path_logprobs[prompt_token_count:])
        else:
            return 0.0  # Trajectory is just the prompt, P=1.0, logP=0.0

    def get_continuation_prob(self, prompt_token_count: int) -> float:
        """
        Get P(continuation | prompt) given prompt token count.

        Args:
            prompt_token_count: Number of prompt tokens (edges from root)

        Returns:
            Probability of continuation after prompt
        """
        return np.exp(self.get_continuation_logprob(prompt_token_count))

    def get_conditional_probabilities(
        self, trajectory_nodes: List[TreeNode], prompt: String, normalize: bool = False
    ) -> np.ndarray:
        """
        Get conditional probabilities P(trajectories | prompt).

        Args:
            trajectory_nodes: List of trajectory nodes
            prompt: The prompt string
            normalize: If True, normalize probabilities to sum to 1

        Returns:
            Array of probabilities (normalized if normalize=True)
        """
        log_probs = []
        for traj_node in trajectory_nodes:
            log_prob = traj_node.get_conditional_logprob(prompt)
            log_probs.append(log_prob)

        # Convert log probs to probabilities with numerical stability
        log_probs_array = np.array(log_probs)
        max_log_prob = np.max(log_probs_array)
        probs = np.exp(log_probs_array - max_log_prob)

        if normalize:
            probs = probs / probs.sum()

        return probs

    def find_node(self, prefix: String) -> Optional[TreeNode]:
        """
        Find node matching a prefix string in this subtree.

        Args:
            prefix: Prefix string to search for

        Returns:
            Node with matching string, or None if not found
        """
        if self.string.tokens == prefix.tokens:
            return self

        for child in self.children.values():
            result = child.find_node(prefix)
            if result:
                return result

        return None

    def find_trajectory_node(self, trajectory: String) -> Optional[TreeNode]:
        """
        Find trajectory node matching a String.

        Args:
            trajectory: String to search for

        Returns:
            TreeNode with matching trajectory, or None if not found
        """
        if self.is_trajectory() and self.string.tokens == trajectory.tokens:
            return self

        for child in self.children.values():
            result = child.find_trajectory_node(trajectory)
            if result:
                return result

        return None

    @property
    def token_id(self) -> Optional[int]:
        """Get token_id from metadata if available."""
        return self.metadata.get("token_id")

    def __repr__(self) -> str:
        dist_info = (
            f"dist={self.next_token_distribution is not None}"
            if self.next_token_distribution is not None
            else "no_dist"
        )
        return (
            f"TreeNode(string={self.string}, "
            f"depth={self.depth()}, "
            f"prob={self.probability():.6f}, "
            f"children={len(self.children)}, "
            f"{dist_info})"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        # Get last token
        last_token = self.string.tokens[-1] if len(self.string.tokens) > 0 else "(root)"

        # Path probability
        path_logprob = self.path_logprob()
        path_prob = self.probability()

        # Distribution info
        if self.next_token_distribution is not None:
            # Compute entropy
            dist = self.next_token_distribution
            mask = dist > 0
            entropy = float(-np.sum(dist[mask] * np.log(dist[mask])))

            # Top tokens
            top_indices = np.argsort(dist)[-5:][::-1]
            top_probs = [(int(idx), float(dist[idx])) for idx in top_indices]
            top_str = ", ".join(f"[{idx}]={p:.4f}" for idx, p in top_probs)
            dist_info = f"entropy={entropy:.3f}, top5: {top_str}"
        else:
            dist_info = "no distribution"

        # Build string
        text = self.string.to_text()
        text_display = f"{text[:80]}..." if len(text) > 80 else text

        lines = [
            "TreeNode:",
            f'  Text: "{text_display}"',
            f"  Tokens: {len(self.string)} | Last: '{last_token}' (id={self.token_id})",
            f"  Depth: {self.depth()} | is_leaf: {self.is_leaf()} | is_trajectory: {self.is_trajectory()}",
            f"  Path prob: {path_prob:.6e} | logprob: {path_logprob:.4f}",
            f"  Children: {len(self.children)} | child_logprobs: {len(self.child_logprobs)}",
            f"  Distribution: {dist_info}",
        ]

        # Children preview
        if self.children:
            child_preview = ", ".join(f"'{k}'" for k in list(self.children.keys())[:5])
            if len(self.children) > 5:
                child_preview += f", ... (+{len(self.children) - 5})"
            lines.append(f"  Child tokens: {child_preview}")

        # Metadata
        if self.metadata:
            meta_str = ", ".join(f"{k}={v}" for k, v in list(self.metadata.items())[:5])
            lines.append(f"  Metadata: {meta_str}")

        return "\n".join(lines)


class LLMTree:
    """
    Singleton tree for each LLM.

    Design:
    - One tree instance per LLM (singleton pattern)
    - Stores only the root node
    - All tree operations work on TreeNode references
    """

    _instances: Dict[str, LLMTree] = {}

    def __init__(self, llm_id: str, tokenizer=None, system: AbstractSystem = None):
        """
        Initialize tree for an LLM.

        Note: Use get_tree() class method instead of direct instantiation.

        Args:
            llm_id: Unique identifier for the LLM
            tokenizer: Tokenizer for this LLM
            system: Optional system for evaluating nodes
        """
        self.llm_id = llm_id
        self.tokenizer = tokenizer
        self.system = system
        self.root = TreeNode(string=String.empty())

    @classmethod
    def get_tree(
        cls,
        llm_id: str,
        tokenizer=None,
        system: AbstractSystem = None,
        reset: bool = False,
    ) -> LLMTree:
        """
        Get or create tree for an LLM (singleton pattern).

        Args:
            llm_id: Unique identifier for the LLM
            tokenizer: Tokenizer for this LLM
            system: Optional system for evaluating nodes
            reset: If True, reset the tree even if it exists

        Returns:
            LLMTree instance for this LLM
        """
        if reset or llm_id not in cls._instances:
            cls._instances[llm_id] = LLMTree(llm_id, tokenizer, system)
        else:
            # Update tokenizer/system if provided
            if tokenizer is not None:
                cls._instances[llm_id].tokenizer = tokenizer
            if system is not None:
                cls._instances[llm_id].system = system

        return cls._instances[llm_id]

    @classmethod
    def clear_tree(cls, llm_id: str) -> bool:
        """
        Clear tree for an LLM.

        Args:
            llm_id: LLM identifier

        Returns:
            True if tree was cleared, False if didn't exist
        """
        if llm_id in cls._instances:
            del cls._instances[llm_id]
            return True
        return False

    @classmethod
    def clear_all_trees(cls):
        """Clear all LLM trees."""
        cls._instances.clear()

    @classmethod
    def list_llms(cls) -> List[str]:
        """Get list of all LLM IDs with trees."""
        return list(cls._instances.keys())

    def get_trajectories(self) -> List[String]:
        """Get all complete trajectories in the tree."""
        return self.root.get_trajectories()

    def total_mass_logprob(self) -> float:
        """Log of total probability mass in the tree."""
        return self.root.branch_mass_logprob()

    def total_mass(self) -> float:
        """Total probability mass in the tree."""
        return self.root.branch_mass()

    def max_depth(self) -> int:
        """Maximum depth of any node in the tree."""
        max_d = 0

        def traverse(node: TreeNode):
            nonlocal max_d
            max_d = max(max_d, node.depth())
            for child in node.children.values():
                traverse(child)

        traverse(self.root)
        return max_d

    def get_node(self, prefix: String) -> Optional[TreeNode]:
        """
        Find node with given prefix (convenience method).

        Args:
            prefix: Prefix string to find

        Returns:
            TreeNode with matching string, or None
        """
        return self.root.find_node(prefix)

    def __repr__(self) -> str:
        return (
            f"LLMTree(llm_id='{self.llm_id}', "
            f"total_mass={self.total_mass():.6f}, "
            f"max_depth={self.max_depth()})"
        )
