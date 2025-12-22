# Refactoring Status

## ✅ COMPLETED

### Core Fixes
1. **AbstractScoreOperator** - Fixed to take `AbstractSystem` and `String`
2. **AbstractDifferenceOperator** - Fixed to take two `AbstractSystem` objects and one `String`
3. **String utilities** - Added `to_text()`, `from_text()`, `from_tokens()` methods
4. **AbstractSystem math operations** - Added:
   - `core()` - compute system core
   - `orientation()` - deviation from core
   - `deviance()` - scalar measure of deviation
   - `expected_deviance()` - average deviation
   - `deviance_variance()` - variance of deviation
   - `core_entropy()` - entropy of core

### Structural Changes
5. **Flattened folder structure**:
   - `implementations/structures/` → `xenotechnics/structures/`
   - `implementations/operators/` → `xenotechnics/operators/`
   - `implementations/systems/` → `xenotechnics/systems/`
   - `implementations/trees/` → `xenotechnics/trees/`

6. **Removed tree abstraction** - Single `Tree` class instead of abstract + implementation
7. **Cleaned base folder** - Removed deprecated files (core.py, statistics.py, structures.py, utils.py)
8. **Removed empty folders** - statistics/, estimation/, control/
9. **Updated all operator implementations** to match new signatures

## Current Structure

```
xenotechnics/
├── README.md                 # ✓
├── paper.pdf                 # ✓
├── schemas.py                # ✓
├── __init__.py               # ✓ Updated
├── common/                   # ✓ Abstract base classes
│   ├── strings.py            # ✓ With utilities
│   ├── structures.py         # ✓
│   ├── systems.py            # ✓ With math operations
│   └── operators.py          # ✓ Fixed signatures
├── structures/               # ✓ Implementations
├── operators/                # ✓ Implementations
├── systems/                  # ✓ Implementations
├── trees/                    # ✓ Simplified
│   └── tree.py               # ✓ Single Tree class
├── data/                     # ✓ Data objects
├── examples/                 # ✓
└── [Files to organize]:
    ├── dynamics.py           # → Move to dynamics/
    ├── homogenization.py     # → Move to estimation/
    └── xeno_reproduction.py  # → Move to control/
```

## 🔄 IN PROGRESS

### Folder Creation
- Created empty folders: estimation/, control/, dynamics/
- Need to organize code into these folders

## 📋 REMAINING TASKS

### High Priority
1. **Create estimation/ folder structure** with homogenization code
2. **Create control/ folder structure** with xenoreproduction code
3. **Create dynamics/ folder structure** with:
   - states.py
   - dynamics.py
   - evaluation.py
4. **Create mid-level system abstractions**:
   - VectorSystem (main paper)
   - GeneralizedStructureSystem
   - ExcessSystem (Appendix A)
   - DeficitSystem (Appendix A)

### Medium Priority
5. **Implement Appendix B singleton structure**
6. **Propose and implement 2D system**
7. **Enhance Tree for dynamic traversal**:
   - Build dynamically during distribution traversal
   - Get child node for greedy path
   - Condition through prompt (return subtree with new root)
8. **Smart tree data storage per LLM**:
   - Separate data for different LLMs
   - Shared node updates across Tree() and Subtree() objects

## Notes

### Operator Signature Changes
- **Old**: `score_op(system, string)` + `diff_op(system, string1, string2)`
- **New**: `score_op(system, string)` + `diff_op(system1, system2, string)`
- This allows comparing different systems on the same string

### Tree Simplification
- **Old**: AbstractTreeNode + AbstractGenerationTree + implementations
- **New**: Single TreeNode and GenerationTree classes
- Works directly with AbstractSystem
- Includes dynamic features: greedy_path(), condition_on(), prune()

### Math in AbstractSystem
All core mathematical operations now in AbstractSystem:
- Statistics (core, orientation, deviance)
- Homogenization metrics (expected_deviance, deviance_variance, core_entropy)
- Implementations just need compliance()

## Next Steps

**Recommended order:**
1. Organize estimation/ folder (homogenization detection)
2. Organize control/ folder (xenoreproduction strategies)
3. Organize dynamics/ folder (trajectory analysis)
4. Create mid-level system abstractions
5. Implement new systems (singleton, 2D)
6. Enhance tree for advanced use cases
