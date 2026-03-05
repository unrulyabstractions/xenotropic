# Trial Configurations

Config files for trajectory generation and scoring experiments.

## Directory Structure

```
trials/
├── generation/     # Generation configs (prompts, models, branches)
└── scoring/        # Scoring configs (judgment questions)
```

## Generation Config

`trials/generation/<name>.json`

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "prompt": "Describe what happened next in two sentences.",
  "trunk": "The ",
  "branches": ["doctor", "criminal", "musician"]
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | required | HuggingFace model ID |
| `prompt` | string | required | Text prompt for generation |
| `trunk` | string | `""` | Common prefix for all branches |
| `branches` | list[str] | `[]` | Branch prefixes (creates groups) |

### Branches

If `branches` is empty or omitted, all trajectories belong to a single "trunk" group.

With branches `["doctor", "criminal", "musician"]` and trunk `"The "`:
- Group 0 (trunk): All trajectories pooled
- Group 1: "The doctor..."
- Group 2: "The criminal..."
- Group 3: "The musician..."

## Scoring Config

`trials/scoring/<name>.json`

```json
{
  "model": "Qwen/Qwen3-4B-Instruct-2507",
  "categorical_judgements": [
    "Does this text involve helping or healing someone?",
    "Does this text involve breaking the law or violence?",
    "Does this text involve music or art?"
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `model` | string | Model for making judgments |
| `categorical_judgements` | list[str] | Yes/No questions to evaluate |

### Categorical Judgments

Each judgment becomes a dimension in the compliance vector Λ(x):
- `1` = Yes (compliant with structure)
- `0` = No (not compliant)
- `null` = Could not determine (treated as 0.5)

## How This Demonstrates Diversity/Core/Deviance

The profession example clearly shows the framework concepts:

**Per-branch cores** represent the "typical" output for each condition:
- doctor: [1, 0, 0] - always involves helping
- criminal: [0, 1, 0] - always involves crime
- musician: [0, 0, 1] - always involves music

**Trunk core** is the average across all branches:
- ~[0.33, 0.33, 0.33] - the "center" of all outputs

**Orientation** θ(x) = Λ(x) - core tells us how each trajectory differs from center:
- A doctor story has orientation ~[0.67, -0.33, -0.33] - leans toward helping

**Deviance** ∂(x) = ||θ(x)|| measures how far from the core:
- Within a branch: low deviance (trajectories are similar)
- Across trunk: high deviance (trajectories spread across all types)
