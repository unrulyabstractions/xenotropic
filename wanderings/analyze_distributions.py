"""
Helper script to load and analyze full token distributions
"""

import json
from pathlib import Path

import numpy as np


def load_distributions():
    """Load the full distributions and metadata"""
    SCRIPT_DIR = Path(__file__).parent
    OUTPUT_DIR = SCRIPT_DIR / "output"

    # Load metadata
    with open(OUTPUT_DIR / "token_distributions_metadata.json") as f:
        metadata = json.load(f)

    # Load full distributions
    data = np.load(OUTPUT_DIR / "token_distributions_full.npz")
    distributions = data["distributions"]  # Shape: (num_steps, vocab_size)

    return distributions, metadata


def analyze_step(distributions, metadata, step_num):
    """Analyze a specific generation step"""
    step_dist = distributions[step_num]  # Shape: (vocab_size,)
    step_meta = metadata["distributions"][step_num]

    print(f"\n{'=' * 60}")
    print(f"Step {step_num}")
    print(f"{'=' * 60}")
    print(f"Sampled token: '{step_meta['sampled_token']}'")
    print(f"Sampled probability: {step_meta['sampled_probability']:.6f}")
    print(f"Entropy: {step_meta['entropy']:.4f}")
    print("\nDistribution stats:")
    print(f"  Min probability: {step_dist.min():.2e}")
    print(f"  Max probability: {step_dist.max():.6f}")
    print(f"  Sum: {step_dist.sum():.6f}")
    print(f"  Non-zero tokens: {np.count_nonzero(step_dist):,}")

    # Show top-10
    top_10_indices = np.argsort(step_dist)[-10:][::-1]
    print("\nTop 10 tokens:")
    for i, idx in enumerate(top_10_indices, 1):
        prob = step_dist[idx]
        print(f"  {i}. Token ID {idx}: p={prob:.6f}")


def compare_steps(distributions, metadata, step1, step2):
    """Compare distributions at two different steps"""
    dist1 = distributions[step1]
    dist2 = distributions[step2]

    # KL divergence from step1 to step2
    kl_div = np.sum(dist1 * np.log((dist1 + 1e-10) / (dist2 + 1e-10)))

    # JS divergence (symmetric)
    m = 0.5 * (dist1 + dist2)
    js_div = 0.5 * np.sum(dist1 * np.log((dist1 + 1e-10) / (m + 1e-10))) + 0.5 * np.sum(
        dist2 * np.log((dist2 + 1e-10) / (m + 1e-10))
    )

    print(f"\n{'=' * 60}")
    print(f"Comparing Step {step1} vs Step {step2}")
    print(f"{'=' * 60}")
    print(f"Step {step1} token: '{metadata['distributions'][step1]['sampled_token']}'")
    print(f"Step {step2} token: '{metadata['distributions'][step2]['sampled_token']}'")
    print(f"\nKL divergence (step{step1}||step{step2}): {kl_div:.4f}")
    print(f"JS divergence: {js_div:.4f}")


if __name__ == "__main__":
    print("📊 Loading distributions...")
    distributions, metadata = load_distributions()

    print("\n✅ Loaded distributions")
    print(f"   Shape: {distributions.shape}")
    print(f"   Model: {metadata['model']}")
    print(f"   Prompt: {metadata['prompt']}")
    print(f"   Generated: {metadata['num_steps']} tokens")

    # Analyze first few steps
    analyze_step(distributions, metadata, step_num=0)
    analyze_step(distributions, metadata, step_num=5)
    analyze_step(distributions, metadata, step_num=10)

    # Compare steps
    compare_steps(distributions, metadata, step1=0, step2=10)

    print("\n" + "=" * 60)
    print("✅ Analysis complete!")
