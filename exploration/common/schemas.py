"""
Schemas for LLM exploration and statistics estimation.

Decomposes StatisticsEstimation into modular dataclasses following the
theoretical framework from the paper:
- LLMTreeInfo: Tree structure representation
- TrajectoryInfo: Trajectory probabilities and metadata
- SystemInfo: System descriptions
- CoreInfo: System cores (expected compliance)
- OrientationInfo: Deviations from cores
- DevianceInfo: Scalar deviance statistics
- StatisticsEstimation: Complete estimation results

Paper Reference:
- Section 3.1: LLMs as trees of strings
- Section 3.3: Structure cores and system cores (Eq. 5, 6)
- Section 3.3: Orientations and deviances (Eq. 7, 8)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np

from xenotechnics.common import AbstractSystem, String
from xenotechnics.trees.tree import TreeNode


# ============================================================================
# Tree Schemas
# ============================================================================


@dataclass
class LLMTreeInfo:
    """
    Information about the LLM generation tree.

    Paper (Section 3.1): "Any LLM induces a tree on Str: the root is ⊥,
    each node is a string, the leaves are trajectories, and the edges
    connect strings to their next-token continuations with probability
    p(t_{p+1}|x_p)."

    Attributes:
        root: Root node of the tree (⊥)
        num_nodes: Total number of nodes in tree
        max_depth: Maximum depth of any trajectory
        num_trajectories: Number of complete trajectories (leaves)
        total_mass: Total probability mass in tree
        total_mass_logprob: Log of total probability mass
    """
    root: Optional[TreeNode] = None
    num_nodes: int = 0
    max_depth: int = 0
    num_trajectories: int = 0
    total_mass: float = 0.0
    total_mass_logprob: float = -np.inf

    @classmethod
    def from_tree(cls, tree: TreeNode) -> LLMTreeInfo:
        """
        Create LLMTreeInfo from a TreeNode.

        Args:
            tree: Root node of the tree

        Returns:
            LLMTreeInfo with computed statistics
        """
        # Count nodes
        def count_nodes(node: TreeNode) -> int:
            if not hasattr(node, 'children'):
                return 1
            count = 1
            for child in node.children.values():
                if isinstance(child, TreeNode) or hasattr(child, 'children'):
                    count += count_nodes(child)
            return count

        # Get max depth
        def get_depth(node: TreeNode) -> int:
            if not hasattr(node, 'children') or not node.children:
                return 0
            child_depths = []
            for child in node.children.values():
                if isinstance(child, TreeNode) or hasattr(child, 'children'):
                    child_depths.append(get_depth(child))
            if not child_depths:
                return 0
            return 1 + max(child_depths)

        # Count trajectories
        trajectory_nodes = tree.get_trajectory_nodes()

        return cls(
            root=tree,
            num_nodes=count_nodes(tree),
            max_depth=get_depth(tree),
            num_trajectories=len(trajectory_nodes),
            total_mass=tree.branch_mass(),
            total_mass_logprob=tree.branch_mass_logprob(),
        )

    def to_dict(self, include_tree: bool = False) -> Dict[str, Any]:
        """
        Convert to JSON-serializable dictionary.

        Args:
            include_tree: If True, include full tree structure (can be large)

        Returns:
            Dictionary representation
        """
        result = {
            "num_nodes": self.num_nodes,
            "max_depth": self.max_depth,
            "num_trajectories": self.num_trajectories,
            "total_mass": float(self.total_mass),
            "total_mass_logprob": float(self.total_mass_logprob),
        }

        if include_tree and self.root is not None:
            # TODO: Implement tree serialization
            result["tree_root_depth"] = self.root.depth()

        return result


# ============================================================================
# Trajectory Schemas
# ============================================================================


@dataclass
class TrajectoryInfo:
    """
    Information about trajectories and their probabilities.

    Paper (Section 3.1): "A trajectory is a string ending with ⊤."
    Paper (Eq. 1): "Σ_{y∈Str⊤(x_p)} p(y|x_p) = 1"

    Attributes:
        trajectories: List of completed trajectory strings
        probabilities: Normalized probabilities (sum to 1)
        log_probabilities: Log probabilities for numerical stability
        num_trajectories: Number of trajectories
        total_probability_mass: Sum of trajectory probabilities
    """
    trajectories: List[String]
    probabilities: np.ndarray
    log_probabilities: np.ndarray
    num_trajectories: int
    total_probability_mass: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "trajectories": [traj.to_text() for traj in self.trajectories],
            "probabilities": self.probabilities.tolist(),
            "log_probabilities": self.log_probabilities.tolist(),
            "num_trajectories": int(self.num_trajectories),
            "total_probability_mass": float(self.total_probability_mass),
        }

    def get_top_k(self, k: int) -> TrajectoryInfo:
        """Get top-k trajectories by probability."""
        top_indices = np.argsort(-self.probabilities)[:k]
        return TrajectoryInfo(
            trajectories=[self.trajectories[i] for i in top_indices],
            probabilities=self.probabilities[top_indices],
            log_probabilities=self.log_probabilities[top_indices],
            num_trajectories=min(k, self.num_trajectories),
            total_probability_mass=float(np.sum(self.probabilities[top_indices])),
        )


# ============================================================================
# System Schemas
# ============================================================================


@dataclass
class SystemInfo:
    """
    Information about a system and its structures.

    Paper (Section 3.2, Eq. 2-3):
    "For a string x ∈ Str, the degree of structure compliance is α_i(x).
    Ideal compliance corresponds to α_i(x) = 1, and no compliance
    corresponds to α_i(x) = 0."

    "We call a system the collection of structures of interest. We define
    the system compliance as a vector of compliances across particular
    structures: Λ_n(x) := (α_1(x), ..., α_n(x))"

    Attributes:
        system: The system object
        system_idx: Index of this system in the list
        system_type: Type name of the system
        system_repr: String representation
        num_structures: Number of structures in the system (n)
        structure_names: Optional names for each structure
    """
    system: AbstractSystem
    system_idx: int
    system_type: str
    system_repr: str
    num_structures: int
    structure_names: Optional[List[str]] = None

    @classmethod
    def from_system(cls, system: AbstractSystem, idx: int) -> SystemInfo:
        """Create SystemInfo from a system."""
        return cls(
            system=system,
            system_idx=idx,
            system_type=type(system).__name__,
            system_repr=str(system),
            num_structures=len(system),
            structure_names=getattr(system, 'questions', None),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result = {
            "system_idx": self.system_idx,
            "type": self.system_type,
            "repr": self.system_repr,
            "num_structures": self.num_structures,
        }
        if self.structure_names:
            result["structure_names"] = self.structure_names
        return result


@dataclass
class CoreInfo:
    """
    Information about system cores (expected compliance).

    Paper (Section 3.3, Eq. 5-6):
    "Structure core: ⟨α_i⟩ = Σ_{y∈Str⊤} p(y)α_i(y)"
    "System core: ⟨Λ_n⟩ = Σ_{y∈Str⊤} p(y)Λ_n(y)"

    The core represents the expected structural compliance under the
    probability distribution defined by the LLM.

    Attributes:
        system_idx: Index of the system this core belongs to
        core_vector: The system core as a vector (α_1, ..., α_n)
        structure_cores: Individual structure cores
    """
    system_idx: int
    core_vector: np.ndarray
    structure_cores: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result = {
            "system_idx": self.system_idx,
            "core_vector": self.core_vector.tolist() if hasattr(self.core_vector, 'tolist') else list(self.core_vector),
        }
        if self.structure_cores:
            result["structure_cores"] = self.structure_cores
        return result

    def to_array(self) -> np.ndarray:
        """Get core as numpy array."""
        if isinstance(self.core_vector, np.ndarray):
            return self.core_vector
        elif hasattr(self.core_vector, 'to_array'):
            return self.core_vector.to_array()
        else:
            return np.array(self.core_vector)


@dataclass
class OrientationInfo:
    """
    Information about orientations (deviations from core).

    Paper (Section 3.3, Eq. 7):
    "The orientation of a given string relative to the given system core is:
    θ_n(x) = Λ_n(x) - ⟨Λ_n⟩"

    "We can think of orientation as a characterization of queerness for a
    string. If the system core tells us what is normatively complied with,
    orientations tell us in what ways a string is non-normative."

    Attributes:
        system_idx: Index of the system
        orientations: Orientation vectors for each trajectory (n x m array)
        mean_orientation: Mean orientation across trajectories
        std_orientation: Standard deviation of orientations
    """
    system_idx: int
    orientations: np.ndarray
    mean_orientation: Optional[np.ndarray] = None
    std_orientation: Optional[np.ndarray] = None

    def __post_init__(self):
        """Compute statistics if not provided."""
        if self.mean_orientation is None:
            self.mean_orientation = np.mean(self.orientations, axis=0)
        if self.std_orientation is None:
            self.std_orientation = np.std(self.orientations, axis=0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "system_idx": self.system_idx,
            "orientations": self.orientations.tolist(),
            "mean_orientation": self.mean_orientation.tolist(),
            "std_orientation": self.std_orientation.tolist(),
        }


@dataclass
class DevianceInfo:
    """
    Information about deviances (scalar measures of non-normativity).

    Paper (Section 3.3, Eq. 8):
    "To summarize non-normativity as a single number, we leverage Equation 4
    to define the deviance: ∥θ_n(x)∥_θ = ∂_n(x) ∈ [0, 1]"

    Paper (Section 4, Eq. 12):
    "Expected deviance: E_{y~p(·|x_p)}[∂_n]"
    "Deviance variance: Var_{y~p(·|x_p)}[∂_n]"

    Attributes:
        system_idx: Index of the system
        deviances: Deviance values for each trajectory
        expected_deviance: Expected deviance (weighted by probability)
        variance_deviance: Variance of deviance
        std_deviance: Standard deviation of deviance
        min_deviance: Minimum deviance
        max_deviance: Maximum deviance
    """
    system_idx: int
    deviances: np.ndarray
    expected_deviance: float
    variance_deviance: float
    std_deviance: float
    min_deviance: float
    max_deviance: float

    @classmethod
    def from_deviances(
        cls,
        system_idx: int,
        deviances: np.ndarray,
        probabilities: Optional[np.ndarray] = None
    ) -> DevianceInfo:
        """
        Create DevianceInfo from deviance values.

        Args:
            system_idx: Index of the system
            deviances: Deviance values
            probabilities: Trajectory probabilities (for weighted statistics)

        Returns:
            DevianceInfo with computed statistics
        """
        if probabilities is not None:
            # Weighted statistics
            expected_dev = float(np.average(deviances, weights=probabilities))
            variance_dev = float(np.average(
                (deviances - expected_dev) ** 2,
                weights=probabilities
            ))
        else:
            # Unweighted statistics
            expected_dev = float(np.mean(deviances))
            variance_dev = float(np.var(deviances))

        return cls(
            system_idx=system_idx,
            deviances=deviances,
            expected_deviance=expected_dev,
            variance_deviance=variance_dev,
            std_deviance=float(np.sqrt(variance_dev)),
            min_deviance=float(np.min(deviances)),
            max_deviance=float(np.max(deviances)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "system_idx": self.system_idx,
            "deviances": self.deviances.tolist(),
            "expected_deviance": float(self.expected_deviance),
            "variance_deviance": float(self.variance_deviance),
            "std_deviance": float(self.std_deviance),
            "min_deviance": float(self.min_deviance),
            "max_deviance": float(self.max_deviance),
        }


# ============================================================================
# Complete Statistics Estimation
# ============================================================================


@dataclass
class StatisticsEstimation:
    """
    Complete results from core estimation exploration.

    Combines all the schemas above to provide a complete picture of:
    - The LLM generation tree structure
    - Trajectories and their probabilities
    - System descriptions
    - System cores (expected compliance)
    - Orientations (deviations from cores)
    - Deviances (scalar summaries)

    Paper Overview:
    - Section 3: Theoretical framework
    - Section 4: Homogenization (minimizing deviance)
    - Section 5: Xeno-reproduction (maximizing diversity)

    Attributes:
        tree_info: Information about the generation tree
        trajectory_info: Trajectories and probabilities
        system_infos: Information about each system
        core_infos: System cores for each system
        orientation_infos: Orientations for each system
        deviance_infos: Deviance statistics for each system
        prompt: Original prompt used for exploration
        metadata: Optional additional metadata
    """
    tree_info: LLMTreeInfo
    trajectory_info: TrajectoryInfo
    system_infos: List[SystemInfo]
    core_infos: List[CoreInfo]
    orientation_infos: Optional[List[OrientationInfo]] = None
    deviance_infos: Optional[List[DevianceInfo]] = None
    prompt: Optional[String] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Legacy attributes for backward compatibility
    @property
    def trajectories(self) -> List[String]:
        """Legacy: Access trajectories."""
        return self.trajectory_info.trajectories

    @property
    def probabilities(self) -> np.ndarray:
        """Legacy: Access probabilities."""
        return self.trajectory_info.probabilities

    @property
    def log_probabilities(self) -> np.ndarray:
        """Legacy: Access log probabilities."""
        return self.trajectory_info.log_probabilities

    @property
    def tree(self) -> Optional[TreeNode]:
        """Legacy: Access tree root."""
        return self.tree_info.root

    @property
    def systems(self) -> List[AbstractSystem]:
        """Legacy: Access systems."""
        return [info.system for info in self.system_infos]

    @property
    def cores(self) -> List:
        """Legacy: Access cores."""
        return [info.core_vector for info in self.core_infos]

    @property
    def deviations(self) -> List[np.ndarray]:
        """Legacy: Access deviations (deviances)."""
        if self.deviance_infos:
            return [info.deviances for info in self.deviance_infos]
        return []

    @property
    def total_probability_mass(self) -> float:
        """Legacy: Access total probability mass."""
        return self.trajectory_info.total_probability_mass

    @property
    def num_trajectories(self) -> int:
        """Legacy: Access number of trajectories."""
        return self.trajectory_info.num_trajectories

    def to_dict(self, include_tree: bool = False) -> Dict[str, Any]:
        """
        Convert to JSON-serializable dictionary.

        Args:
            include_tree: Whether to include full tree structure

        Returns:
            Complete dictionary representation
        """
        result = {
            "tree_info": self.tree_info.to_dict(include_tree=include_tree),
            "trajectory_info": self.trajectory_info.to_dict(),
            "system_infos": [info.to_dict() for info in self.system_infos],
            "core_infos": [info.to_dict() for info in self.core_infos],
        }

        if self.orientation_infos:
            result["orientation_infos"] = [
                info.to_dict() for info in self.orientation_infos
            ]

        if self.deviance_infos:
            result["deviance_infos"] = [
                info.to_dict() for info in self.deviance_infos
            ]

        if self.prompt:
            result["prompt"] = self.prompt.to_text()

        if self.metadata:
            result["metadata"] = self.metadata

        return result

    @classmethod
    def from_legacy(
        cls,
        trajectories: List[String],
        probabilities: np.ndarray,
        log_probabilities: np.ndarray,
        tree: TreeNode,
        systems: List[AbstractSystem],
        cores: List,
        deviations: List[np.ndarray],
        total_probability_mass: float,
        num_trajectories: int,
        prompt: String,
    ) -> StatisticsEstimation:
        """
        Create StatisticsEstimation from legacy format.

        This factory method allows creating the new decomposed format
        from the old flat structure for backward compatibility.
        """
        # Create tree info
        tree_info = LLMTreeInfo.from_tree(tree)

        # Create trajectory info
        trajectory_info = TrajectoryInfo(
            trajectories=trajectories,
            probabilities=probabilities,
            log_probabilities=log_probabilities,
            num_trajectories=num_trajectories,
            total_probability_mass=total_probability_mass,
        )

        # Create system infos
        system_infos = [
            SystemInfo.from_system(system, idx)
            for idx, system in enumerate(systems)
        ]

        # Create core infos
        core_infos = []
        for idx, core in enumerate(cores):
            if hasattr(core, 'to_array'):
                core_vector = core.to_array()
            else:
                core_vector = np.array(core) if not isinstance(core, np.ndarray) else core

            core_infos.append(CoreInfo(
                system_idx=idx,
                core_vector=core_vector,
            ))

        # Create deviance infos
        deviance_infos = [
            DevianceInfo.from_deviances(idx, dev, probabilities)
            for idx, dev in enumerate(deviations)
        ]

        return cls(
            tree_info=tree_info,
            trajectory_info=trajectory_info,
            system_infos=system_infos,
            core_infos=core_infos,
            deviance_infos=deviance_infos,
            prompt=prompt,
        )
