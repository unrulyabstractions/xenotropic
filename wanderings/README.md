# Wanderings

Experimental scripts and exploratory code for xenoreproduction research.

## Directory Structure

- `data/` - Downloaded datasets and input data
- `output/` - Generated analysis, reports, and results

## simple_test.py

A script to generate text with Hugging Face transformers while tracking token distributions at each generation step. Uses `model.generate()` with `output_scores=True` to capture full probability distributions.

### Prerequisites

Make sure your virtual environment is set up:

```bash
# From the project root directory
uv sync
```

### Running the Script

```bash
# From wanderings/ directory:
uv run python simple_test.py

# From project root:
uv run python wanderings/simple_test.py
```

### What it does

- Loads **Qwen2.5-0.5B-Instruct** from Hugging Face
- Generates text using `model.generate()` with `output_scores=True`
- **Captures FULL probability distributions** (all vocab tokens) at each step
- Saves distributions to compressed numpy format (~29 MB for 50 steps)
- Saves metadata with top-K tokens to JSON
- Uses Apple Silicon MPS for GPU acceleration

### Output Data

The script saves **two files**:

1. **`token_distributions_metadata.json`** - Human-readable metadata:
   - Prompt and model info
   - Generated text
   - Per-step metadata with top-K tokens
   - Sampled probabilities and entropy

2. **`token_distributions_full.npz`** - Complete probability distributions:
   - Full distribution for **all vocabulary tokens** at each step
   - Shape: `(num_steps, vocab_size)` e.g., `(50, 151936)`
   - Compressed numpy format (~29 MB for 50 steps)
   - Use `np.load()` to access

### First Run

The first time you run the script, it will download the model (~300MB). Subsequent runs will use the cached model.

### Using Different Models

Edit `simple_test.py` and change the `model_name` variable:

```python
# Small & fast (default)
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# Medium size
model_name = "meta-llama/Llama-3.2-3B-Instruct"

# Larger model
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
```

Browse available models at: https://huggingface.co/models

### Customizing the Prompt

Edit the `prompt` variable in the script:

```python
prompt = "Your custom prompt here"
```

### Parameters

Adjust generation parameters in the `generate_with_distributions()` function:

- `max_new_tokens`: Maximum tokens to generate (default: 50)
- `temperature`: Sampling temperature for randomness (default: 0.7)
- `top_k`: Number of top tokens to track per step (default: 10)

### Analyzing Distributions

Use the included `analyze_distributions.py` helper script:

```bash
uv run python wanderings/analyze_distributions.py
```

Or load the data yourself:

```python
import numpy as np
import json

# Load full distributions
data = np.load("output/token_distributions_full.npz")
distributions = data["distributions"]  # Shape: (num_steps, vocab_size)

# Load metadata
with open("output/token_distributions_metadata.json") as f:
    metadata = json.load(f)

# Access distribution at step i
step_dist = distributions[i]  # All vocab_size probabilities
sampled_token_id = metadata["distributions"][i]["sampled_token_id"]
```

### Example Analysis Output

```
Step 0: ' Xen' (p=0.9995)
Distribution stats:
  Max probability: 0.999512
  Non-zero tokens: 41

Step 5: ' reproductive' (p=0.4382)
Distribution stats:
  Max probability: 0.438232
  Non-zero tokens: 1,362
```

---

## hivemind.py

Explores the **Artificial Hivemind** multi-agent conversation datasets from Hugging Face.

### Running the Script

```bash
# From wanderings/ directory:
uv run python hivemind.py

# From project root:
uv run python wanderings/hivemind.py
```

### What it does

1. **Downloads** the `infinite-chats-eval` dataset (100 conversation prompts)
2. **Explores** dataset structure and content
3. **Saves samples** to `output/hivemind_samples.json`
4. **Analyzes** examples using MLX LLM
5. **Generates** analysis report in `output/hivemind_analysis.txt`
6. **Creates** summary statistics in `output/dataset_summary.txt`

### Available Datasets

The script can work with any dataset from the collection:

- `liweijiang/infinite-chats-taxonomy` (26.1k items) - Full taxonomy
- `liweijiang/infinite-chats-eval` (100 items) - Evaluation set (default)
- `liweijiang/infinite-chats-human-absolute` (750 items) - Human ratings
- `liweijiang/infinite-chats-human-pairwise` (500 items) - Pairwise comparisons

Change the dataset in `hivemind.py`:

```python
dataset = load_hivemind_data("liweijiang/infinite-chats-taxonomy")
```

### Outputs

All outputs are saved to the `output/` directory:
- `hivemind_samples.json` - Sample conversations
- `hivemind_analysis.txt` - LLM analysis
- `dataset_summary.txt` - Statistical summary

### Source

Collection: https://huggingface.co/collections/liweijiang/artificial-hivemind
