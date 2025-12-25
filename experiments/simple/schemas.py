"""
Schema definitions for simple experiment.

Uses param_id (from SchemaClass.get_id()) to uniquely identify experiments
based on their parameter values rather than arbitrary experiment_id strings.
"""

from dataclasses import dataclass, field
from typing import Optional

from xenotechnics.common import SchemaClass


@dataclass
class GenerationConfig(SchemaClass):
    """Configuration for trajectory generation."""

    model: str
    base_prompt: str
    branching_points: list[str] = field(default_factory=list)
    temperature: float = 1.8
    top_p: float = 0.995
    top_k: int = 500
    estimation_temperature: float = 0.8
    seed: int = 42


@dataclass
class EstimationConfig(SchemaClass):
    """Configuration for core estimation."""

    model: str
    systems: list[str]
    structures: list[str]


@dataclass
class Params(SchemaClass):
    """
    Full experiment parameters.

    The param_id is derived from get_id() which hashes all parameter values.
    This ensures experiments with identical parameters get the same ID,
    and different parameters get different IDs.
    """

    experiment_id: str  # Human-readable name
    generation: GenerationConfig
    estimation: EstimationConfig

    @property
    def param_id(self) -> str:
        """Get unique parameter-based ID (first 12 chars of hash)."""
        return self.get_id()[:12]

    @property
    def output_dir_name(self) -> str:
        """Get directory name for outputs: {experiment_id}_{param_id}."""
        return f"{self.experiment_id}_{self.param_id}"


@dataclass
class TrajectoryRecord(SchemaClass):
    """Single trajectory with its probability."""

    text: str
    tokens: list[str]  # Token strings, not IDs
    probability: float
    log_probability: float


@dataclass
class GenerationOutput(SchemaClass):
    """Output from a single generation run."""

    param_id: str  # Unique parameter-based ID
    experiment_id: str  # Human-readable name
    prompt_variant: str  # "base", "branch1", etc.
    prompt_text: str
    model: str
    timestamp: str
    min_prob_mass: float
    total_mass: float
    num_trajectories: int
    trajectories: list[TrajectoryRecord]


@dataclass
class StructureResult(SchemaClass):
    """Results for a single structure across all trajectories."""

    structure: str  # The structure query/description
    scores: list[float]  # Score for each trajectory
    core: float  # Computed core value
    expected_deviance: float
    var_deviance: float


@dataclass
class SystemResult(SchemaClass):
    """Results for a single system across all structures."""

    system: str  # System name
    structures: list[StructureResult]
    aggregate_core: float  # Aggregate core across structures
    aggregate_deviance: float


@dataclass
class CoreEstimationOutput(SchemaClass):
    """Output from core estimation."""

    param_id: str  # Unique parameter-based ID
    experiment_id: str  # Human-readable name
    timestamp: str
    judge_model: str
    prompt_variant: str  # Which prompt variant this is for
    prompt_text: str
    num_trajectories: int
    total_mass: float
    systems: list[SystemResult]


@dataclass
class TrajectoryDynamics(SchemaClass):
    """Dynamics information for a single trajectory."""

    trajectory_idx: int
    text: str
    probability: float
    structure_scores: dict[str, float]  # {structure: score}
    system_deviances: dict[str, float]  # {system: deviance from core}
    is_best_for_structure: dict[str, bool]  # {structure: True/False}


@dataclass
class VisualizationOutput(SchemaClass):
    """Output from visualization generation."""

    param_id: str
    experiment_id: str
    timestamp: str
    prompt_variant: str
    tree_image_path: str
    dynamics_image_path: Optional[str] = None
    description: str = ""
    llm_analysis: str = ""
