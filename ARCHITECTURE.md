# Xenotechnics Architecture

## Clean, Intuitive Structure

The codebase is organized into **5 clear folders**:

```
xenotechnics/
├── common/                 # Abstract base classes (what things ARE)
├── implementations/        # Mathematical objects (HOW things work)
├── data/                   # Data structures (scores, states, metrics)
├── compute/                # Analytical computations (cores, deviances, homogenization)
└── strategies/             # Intervention strategies (xenoreprod)
```

## Detailed Structure

```
xenotechnics/
├── common/                 # 🎯 Abstract interfaces
│   ├── strings.py          # String, Trajectory
│   ├── structures.py       # AbstractStructure
│   ├── systems.py          # AbstractSystem
│   ├── operators.py        # AbstractScoreOperator, AbstractDifferenceOperator
│   └── trees.py            # AbstractTreeNode, AbstractGenerationTree
│
├── implementations/        # ⚙️  Mathematical implementations
│   ├── structures/
│   │   ├── basic.py        # Length, TokenSet, Pattern, Sequence
│   │   ├── statistical.py  # Entropy, Repetition
│   │   └── composite.py    # Functional, Composite
│   ├── operators/
│   │   ├── main.py         # L2, L1, Linf, Mean (main paper)
│   │   └── appendix.py     # EscortPowerMean, RenyiEntropy (Appendix A)
│   ├── systems/
│   │   └── basic.py        # System
│   └── trees/
│       └── basic.py        # TreeNode, GenerationTree
│
├── data/                   # 📦 Pure data structures
│   ├── scores.py           # InterventionScores
│   ├── states.py           # DynamicsState, ConditionalStatistics
│   └── metrics.py          # HomogenizationMetrics
│
├── compute/                # 📊 Analytical computations
│   ├── cores.py            # structure_core(), system_core()
│   ├── orientations.py     # orientation()
│   ├── deviances.py        # deviance(), expected_deviance(), variance()
│   ├── dynamics.py         # TrajectoryDynamics, track_generation_dynamics()
│   └── homogenization.py   # compute_homogenization_metrics(), diagnostics
│
└── strategies/             # 🎛️  Intervention strategies
    ├── xenoreprod.py       # xeno_reproduction_distribution/trajectory()
    └── scoring.py          # score_diversity(), score_fairness(), score_constraints()
```

## Design Principles

### 1. **Clear Naming Convention**
- **Abstract base classes**: `AbstractX` (e.g., `AbstractStructure`, `AbstractSystem`)
- **Main implementations**: Clean names (e.g., `System`, `GenerationTree`)
- **Specific variants**: Descriptive names (e.g., `LengthStructure`, `L2ScoreOperator`)

### 2. **Separation by Purpose**
- **common/** = interfaces and protocols
- **implementations/** = mathematical objects
- **data/** = data classes with minimal logic
- **compute/** = pure computations (input → output)
- **strategies/** = intervention algorithms

### 3. **Paper Alignment**
Each module maps directly to paper sections:
- `common/strings.py` → Section 3.1 (Strings and Trajectories)
- `common/trees.py` → Section 3.1 (LLMs as trees of strings)
- `common/structures.py` → Section 3.2 (Structures)
- `implementations/operators/main.py` → Section 3.2 (Score operators)
- `implementations/operators/appendix.py` → Appendix A (Generalized operators)
- `compute/cores.py` → Section 3.3 (Statistical cores)
- `compute/deviances.py` → Section 3.4 (Orientations and deviances)
- `compute/dynamics.py` → Section 3.5 (Trajectory dynamics)
- `compute/homogenization.py` → Section 4 (Homogenization detection)
- `strategies/xenoreprod.py` → Section 5 (Xeno-reproduction)

## Usage Examples

### Basic Usage
```python
from xenotechnics.common import String, AbstractStructure
from xenotechnics.implementations import System, LengthStructure
from xenotechnics.implementations.operators import L2ScoreOperator

# Create system
system = System([LengthStructure(5, 15)])
string = String(('⊥', 'h', 'e', 'l', 'l', 'o', '⊤'))

# Use L2 operator
operator = L2ScoreOperator()
score = operator(system, string)
```

### Using Appendix Operators
```python
from xenotechnics.implementations.operators import EscortPowerMeanOperator, MaxExcessOperator

# Escort power mean (Appendix A.1)
escort_op = EscortPowerMeanOperator(q=2.0, r=1.0)
score = escort_op(system, string)

# Maximum excess (Appendix A.2)
excess_op = MaxExcessOperator()
diff = excess_op(system, string1, string2)
```

### Working with Data Objects
```python
from xenotechnics.data import InterventionScores, HomogenizationMetrics

# Scores are just data
scores = InterventionScores(
    diversity_score=0.8,
    fairness_score=0.6,
    constraint_score=0.4
)
total = scores.total(lambda_d=1.0, lambda_f=0.5, lambda_c=0.3)

# Metrics are just data
metrics = HomogenizationMetrics(
    expected_deviance=0.1,
    deviance_variance=0.05,
    core_entropy=1.2,
    core=np.array([0.5, 0.3, 0.2])
)
is_bad = metrics.is_homogenized()
```

### Working with Trees
```python
from xenotechnics.implementations import GenerationTree

# Build generation tree
tree = GenerationTree()
root = tree.root

# Add branches
hello_node = root.add_child("Hello", probability=0.6)
hello_node.add_child("world", probability=0.4)
hello_node.children[-1].add_child("⊤", probability=0.4)

# Calculate statistics
trajectories = tree.get_trajectories()
total_mass = tree.total_mass()
branch_mass = tree.branch_mass(hello_node)
coverage = tree.coverage([hello_node])  # Fraction of mass in this branch

# Prune low-probability branches
pruned_tree = tree.prune(min_probability=0.1)
```

## Folder Responsibilities

| Folder | Responsibility | Examples |
|--------|---------------|----------|
| `common/` | Define interfaces | `AbstractStructure`, `AbstractSystem`, `AbstractScoreOperator` |
| `implementations/` | Mathematical objects | `System`, `LengthStructure`, `L2ScoreOperator`, `GenerationTree` |
| `data/` | Data structures | `InterventionScores`, `DynamicsState`, `HomogenizationMetrics` |
| `compute/` | Pure computation | `system_core()`, `deviance()`, `compute_homogenization_metrics()` |
| `strategies/` | Interventions | `xenoreprod`, scoring functions |

## Benefits of This Structure

1. **Clear Mental Model**:
   - Looking for interface? → `common/`
   - Want implementation? → `implementations/`
   - Need to compute something? → `compute/`
   - Want to intervene? → `strategies/`

2. **Easy to Extend**:
   - New structure? Add to `implementations/structures/`
   - New operator? Add to `implementations/operators/`
   - Want Appendix B formulation? Add `implementations/operators/appendix_b.py`
   - New intervention? Add to `strategies/`

3. **Testing is Obvious**:
   - Test abstractions? → Interface contracts in `common/`
   - Test implementations? → Concrete behavior in `implementations/`
   - Test computations? → Accuracy in `compute/`
   - Test strategies? → Effectiveness in `strategies/`

4. **Paper Transparency**:
   - Main paper = `implementations/operators/main.py`
   - Appendix A = `implementations/operators/appendix.py`
   - Future appendices = new files in same pattern
   - Clear mapping from paper sections to code modules

## Migration Path

Old code can gradually migrate:
1. Update imports: `from xenotechnics.common import AbstractStructure`
2. Use new class names: `System` instead of `BasicSystem`
3. Switch operators: `from xenotechnics.implementations.operators.appendix import EscortPowerMeanOperator`

## Implementation Status

- ✅ All common abstractions
- ✅ All structure implementations
- ✅ All operator implementations (main + appendix)
- ✅ All system implementations
- ✅ All tree implementations
- ✅ All data objects
- ⏳ Compute functions (cores, orientations, deviances, dynamics, homogenization)
- ⏳ Intervention strategies (xenoreprod, scoring)

The structure is clean and ready - just need to implement the compute and strategies modules!
