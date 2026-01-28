# Exploration Module

The exploration module provides tools for trajectory collection, core estimation, and model interaction for language model analysis.

## Quick Start

```python
from exploration import ModelRunner, TrajectoryCollector, CoreEstimator

# Load model
runner = ModelRunner("Qwen/Qwen2.5-0.5B-Instruct")

# Collect trajectories
collector = TrajectoryCollector(runner)
result = collector.collect("Once upon a time")

print(f"Collected {len(result.trajectories)} trajectories")
print(f"Total mass: {result.total_mass:.4f}")

# Estimate core
from xenotechnics.systems import JudgeVectorSystem
system = JudgeVectorSystem(
    questions=["Is this creative?", "Is this coherent?"],
    model_runner=runner,
)
estimator = CoreEstimator(system)
core_result = estimator.estimate(result.trajectories)

print(f"Expected deviance: {core_result.expected_deviance:.4f}")
```

## Components

### ModelRunner

TransformerLens wrapper for model loading and inference.

```python
from exploration import ModelRunner

# Auto-detects device (MPS > CUDA > CPU) and dtype
runner = ModelRunner("gpt2")

# Explicit configuration
runner = ModelRunner(
    "Qwen/Qwen2.5-0.5B-Instruct",
    device="cuda",
    dtype=torch.float16,
)

# Generate text
output = runner.generate("Hello, world!", max_new_tokens=50)

# Get next token logits
input_ids = runner.tokenize("Hello")
logits, _ = runner.get_next_token_logits(input_ids)

# Run with activation cache
logits, cache = runner.run_with_cache("Hello world")
# cache["blocks.0.hook_resid_post"] -> activations tensor

# Get activation names
names = runner.get_activation_names(
    layers=[0, -1],  # First and last layer
    components=["resid_post", "mlp_out"],
)
```

### TrajectoryCollector

Collects trajectories by sampling from the model.

```python
from exploration import TrajectoryCollector, TrajectoryCollectorConfig

# Configure collection
config = TrajectoryCollectorConfig(
    max_new_tokens=50,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    target_mass=0.95,  # Stop when 95% probability mass covered
    max_iterations=500,
    seed=42,
    # Optional: save model activations
    save_activations=True,
    activation_layers=[-1],  # Last layer only
)

collector = TrajectoryCollector(runner, config)

# Batch collection
result = collector.collect("Once upon a time")
for traj in result.trajectories:
    print(f"{traj.text}: p={traj.probability:.4f}")

# Access statistics
print(f"Collection time: {result.stats.total_time_seconds:.2f}s")
print(f"Trajectories/sec: {result.stats.trajectories_per_second:.2f}")

# Online/streaming collection
for traj in collector.collect_iterator("Hello"):
    print(f"Found: {traj.text}")
    if some_condition:
        break

# With progress callback
def on_progress(progress):
    print(f"Progress: {progress.progress_percent:.1f}%")

result = collector.collect("Hello", progress_callback=on_progress)
```

### CoreEstimator

Estimates cores from collected trajectories using vector systems.

```python
from exploration import CoreEstimator, CoreEstimatorConfig
from xenotechnics.systems import JudgeVectorSystem

# Create system
system = JudgeVectorSystem(
    questions=[
        "Is this text helpful?",
        "Is this text safe?",
    ],
    model_runner=runner,
)

# Create estimator
config = CoreEstimatorConfig(
    q=1.0,  # Escort parameter for probability weighting
    r=1.0,  # Escort parameter for core computation
)
estimator = CoreEstimator(system, config)

# Batch estimation
result = estimator.estimate(trajectories)
print(f"Core: {result.core_vector}")
print(f"E[d]: {result.expected_deviance:.4f}")
print(f"Var[d]: {result.variance_deviance:.4f}")

# Online estimation
running_core = None
running_mass = 0.0

for traj in collector.collect_iterator("Hello"):
    running_core, running_mass = estimator.estimate_online(
        traj, running_core, running_mass
    )
    print(f"Running core: {running_core}, mass: {running_mass:.4f}")
```

### Simple Generators

Clean generators for text generation without tree-building complexity.

```python
from exploration.generators import (
    SimpleGreedyGenerator,
    SimpleSamplingGenerator,
    GenerationConfig,
)

config = GenerationConfig(
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    seed=42,
)

# Greedy generation
greedy = SimpleGreedyGenerator(runner, config)
result = greedy.generate("Hello")
print(f"Generated: {result.text}")
print(f"Log prob: {result.total_logprob:.4f}")

# Sampling generation
sampler = SimpleSamplingGenerator(runner, config)
result = sampler.generate("Hello")
```

## Data Classes

### CollectedTrajectory

```python
@dataclass
class CollectedTrajectory:
    text: str                      # Generated text
    tokens: tuple                  # Individual tokens
    token_ids: tuple               # Token IDs
    probability: float             # Total probability
    log_probability: float         # Total log probability
    per_token_logprobs: List[float]  # Per-token log probs
    activations: Optional[dict]    # Model activations (if enabled)
```

### CollectionResult

```python
@dataclass
class CollectionResult:
    trajectories: List[CollectedTrajectory]
    total_mass: float
    iterations: int
    stats: Optional[CollectionStats]

    @property
    def probabilities(self) -> np.ndarray:
        """Get array of trajectory probabilities."""
```

### CollectionStats

```python
@dataclass
class CollectionStats:
    total_iterations: int
    unique_trajectories: int
    duplicate_trajectories: int
    failed_generations: int
    total_time_seconds: float
    avg_trajectory_length: float
    min_probability: float
    max_probability: float
    stop_reason: str  # "target_mass", "max_iterations", "no_progress"
```

### CoreEstimationResult

```python
@dataclass
class CoreEstimationResult:
    core_vector: np.ndarray
    expected_deviance: float
    variance_deviance: float
    per_trajectory_deviances: List[float]

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
```

## Full Example

```python
"""Complete example: collect trajectories and estimate diversity."""

from exploration import (
    ModelRunner,
    TrajectoryCollector,
    TrajectoryCollectorConfig,
    CoreEstimator,
)
from xenotechnics.systems import JudgeGeneralizedSystem

# 1. Load model
runner = ModelRunner("Qwen/Qwen2.5-0.5B-Instruct")

# 2. Configure collection
collector_config = TrajectoryCollectorConfig(
    max_new_tokens=50,
    temperature=1.0,
    target_mass=0.90,
    seed=42,
)
collector = TrajectoryCollector(runner, collector_config)

# 3. Collect trajectories
print("Collecting trajectories...")
result = collector.collect("The future of AI is")

print(f"Found {len(result.trajectories)} trajectories")
print(f"Total mass: {result.total_mass:.4f}")
print(f"Time: {result.stats.total_time_seconds:.2f}s")

# 4. Create evaluation system
system = JudgeGeneralizedSystem(
    questions=[
        "Is this prediction optimistic?",
        "Is this prediction realistic?",
        "Is this prediction detailed?",
    ],
    model_runner=runner,
    q=1.0,
    r=1.0,
)

# 5. Estimate core
estimator = CoreEstimator(system)
core_result = estimator.estimate(result.trajectories)

print(f"\nCore vector: {core_result.core_vector}")
print(f"Expected deviance: {core_result.expected_deviance:.4f}")
print(f"Variance: {core_result.variance_deviance:.4f}")

# 6. Show trajectories
print("\nTop trajectories:")
sorted_trajs = sorted(
    result.trajectories,
    key=lambda t: t.probability,
    reverse=True,
)
for i, traj in enumerate(sorted_trajs[:5]):
    print(f"  {i+1}. p={traj.probability:.4f}: {traj.text[:50]}...")
```
