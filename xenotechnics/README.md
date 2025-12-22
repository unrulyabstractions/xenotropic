# Xenotechnics: Structure-aware Diversity Pursuit

Implementation of the theoretical framework from the paper:
**"Structure-aware Diversity Pursuit as AI Safety strategy against Homogenization"**

## Overview

This package implements a comprehensive framework for measuring, detecting, and pursuing diversity in LLM outputs. The core insight is that **diversity is always relative to context**, and this context must be made explicit through **structures**.

## Key Concepts

### 1. **Strings and Trajectories** (Section 3.1)
- **String**: A finite sequence of tokens beginning with ⊥ (start-of-sequence)
- **Trajectory**: A string ending with ⊤ (end-of-sequence)

### 2. **Structures** (Section 3.2)
A **structure** specifies a type of organization among tokens. Examples:
- Length constraints
- Token patterns
- Entropy/diversity
- Repetition patterns

Each structure defines a **compliance function** α_i(x) ∈ [0,1]:
- 1.0 = ideal compliance
- 0.0 = no compliance

### 3. **Systems**
A **system** is a collection of structures defining a context for diversity.
System compliance: **Λ_n(x) = (α_1(x), ..., α_n(x))**

### 4. **Cores** (Section 3.3)
The **structure core** ⟨α_i⟩ is the expected structural compliance:
```
⟨α_i⟩ = Σ p(y)·α_i(y)
```

The **system core** ⟨Λ_n⟩ tells us what is **normatively complied with**.

### 5. **Orientations** (Section 3.3)
The **orientation** θ_n(x) measures deviation from the core:
```
θ_n(x) = Λ_n(x) - ⟨Λ_n⟩
```

Orientation characterizes **queerness** - how a string is non-normative.

### 6. **Deviances** (Section 3.3)
The **deviance** ∂_n(x) is a scalar measure of non-normativity:
```
∂_n(x) = ||θ_n(x)||_θ
```

### 7. **Homogenization** (Section 4)
Homogenization is detected when:
- Expected deviance → 0: `E[∂_n] → 0`
- Deviance variance → 0: `Var[∂_n] → 0`
- Core entropy → 0: `H(⟨Λ_n⟩) → 0` (few structures dominate)

### 8. **Xeno-reproduction** (Section 5)
**Xeno-reproduction** is the strategy to combat homogenization through structure-aware diversity pursuit.

Two formulations:
- **Distribution-level**: Search over interventions w ~ π(w) ∝ e^{ρ_χ(w)}
- **Trajectory-level**: Reward individual outputs based on deviance, fairness, and constraints

## Module Organization

```
xenotechnics/
├── core.py              # String, Trajectory, Structure, System
├── structures.py        # Concrete structure implementations
├── statistics.py        # Cores, orientations, deviances
├── dynamics.py          # Trajectory evolution tracking
├── homogenization.py    # Detection and measurement
├── xeno_reproduction.py # Diversity pursuit strategies
├── schemas.py           # Original SchemaClass (preserved)
└── utils.py            # Utilities
```

## Quick Start

```python
import xenotechnics as xeno
import numpy as np

# 1. Create trajectories
trajectories = [
    xeno.create_trajectory("hello world"),
    xeno.create_trajectory("hi"),
    xeno.create_trajectory("greetings everyone"),
]

# 2. Define structures
structures = [
    xeno.LengthStructure(min_length=5, max_length=20),
    xeno.EntropyStructure(),
    xeno.RepetitionStructure(),
]

# 3. Create a system
system = xeno.System(structures)

# 4. Compute compliance
for traj in trajectories:
    compliance = system.compliance(traj)
    print(f"{traj}: {compliance}")

# 5. Compute core and detect homogenization
probs = np.array([0.7, 0.2, 0.1])  # Concentrated distribution
metrics = xeno.compute_homogenization_metrics(system, trajectories, probs)
print(f"Homogenization score: {metrics.homogenization_score()}")
print(f"Is homogenized? {metrics.is_homogenized()}")

# 6. Score an intervention
baseline_core = xeno.system_core(system, trajectories, probs)
diverse_probs = np.array([0.33, 0.33, 0.34])  # More diverse
diverse_core = xeno.system_core(system, trajectories, diverse_probs)

scores = xeno.score_intervention(
    system=system,
    baseline_core=baseline_core,
    baseline_trajectories=trajectories,
    baseline_probs=probs,
    intervention_core=diverse_core,
    intervention_trajectories=trajectories,
    intervention_probs=diverse_probs,
)
print(f"Diversity score: {scores.diversity_score}")
```

## Implemented Structures

- **LengthStructure**: Compliance based on string length
- **TokenSetStructure**: Presence/absence of specific tokens
- **PatternStructure**: Regex pattern matching
- **SequenceStructure**: Sequential token patterns
- **EntropyStructure**: Token diversity (entropy)
- **RepetitionStructure**: Penalizes repetition
- **FunctionalStructure**: Custom compliance function
- **CompositeStructure**: Combine multiple structures

## Core Functions

### Statistics
- `structure_core()` - Compute ⟨α_i⟩
- `system_core()` - Compute ⟨Λ_n⟩
- `orientation()` - Compute θ_n(x)
- `deviance()` - Compute ∂_n(x)
- `expected_deviance()` - Compute E[∂_n]
- `deviance_variance()` - Compute Var[∂_n]
- `core_entropy()` - Compute H(⟨Λ_n⟩)

### Homogenization
- `compute_homogenization_metrics()` - All metrics
- `compare_homogenization()` - Compare distributions
- `diagnose_mode_collapse()` - Diagnostic information

### Xeno-reproduction
- `score_diversity()` - ρ_d: exploration + divergence
- `score_fairness()` - ρ_f: invert ordering + evenness
- `score_constraints()` - ρ_c: target/avoid/conserve
- `score_intervention()` - Combined ρ_χ score
- `trajectory_reward()` - Per-trajectory reward

## Examples

See `examples/simple_example.py` for a complete working example demonstrating:
1. Creating trajectories and structures
2. Computing compliance and cores
3. Detecting homogenization
4. Scoring interventions
5. Comparing distributions

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{anonymous2025xenotechnics,
  title={Structure-aware Diversity Pursuit as AI Safety strategy against Homogenization},
  author={Anonymous Authors},
  journal={Technical AI Safety Conference (TAIS)},
  year={2025}
}
```

## Implementation Notes

This implementation follows the theoretical framework from the paper as closely as possible:
- All equation numbers reference the original paper
- Variable names match the paper's notation (e.g., α_i, Λ_n, θ_n, ∂_n)
- Comments include direct quotes from the paper
- The structure is organized by paper sections

## Future Directions

From the paper (Section 7):
- Specification of structures: Develop taxonomy of structure types
- Computational tractability: Efficient approximation methods
- Operationalization: Tractable, readily applicable formulations
- Evaluation: Connect to existing diversity metrics
- Dynamics investigation: Track bifurcation points, chain-of-thought monitoring
- Ethical analysis: Community participation, consent-based approaches

## License

MIT License (matching the repository)
