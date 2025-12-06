"""Synthetic data generation for testing."""

from __future__ import annotations

import numpy as np
from schemas import TrajectoryRecord


class SyntheticGenerator:
    """Generates fake trajectories with Zipf-like probability distribution."""

    # Various continuation patterns for diverse trees
    WORD_POOLS = {
        "adj": [
            "beautiful",
            "red",
            "pretty",
            "delicate",
            "lovely",
            "amazing",
            "perfect",
            "wonderful",
            "stunning",
            "bright",
            "dark",
            "small",
            "large",
            "tiny",
            "huge",
            "old",
            "young",
            "wise",
            "brave",
        ],
        "noun": [
            "flower",
            "garden",
            "house",
            "tree",
            "bird",
            "cat",
            "dog",
            "river",
            "mountain",
            "sky",
            "sun",
            "moon",
            "star",
            "child",
            "woman",
            "man",
            "king",
            "queen",
            "knight",
            "dragon",
        ],
        "verb": [
            "grew",
            "lived",
            "walked",
            "ran",
            "flew",
            "sang",
            "danced",
            "slept",
            "dreamed",
            "loved",
            "feared",
            "found",
            "lost",
        ],
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate(self, prompt: str, n: int) -> tuple[list[TrajectoryRecord], float]:
        """Generate n trajectories with varied branching patterns."""
        continuations = self._generate_continuations(n)
        probs = self._zipf_probs(len(continuations))

        trajectories = []
        mass = 0.0

        for i, cont in enumerate(continuations):
            p = float(probs[i])
            mass += p
            trajectories.append(
                TrajectoryRecord(
                    text=prompt + cont,
                    probability=p,
                    log_probability=float(np.log(p + 1e-10)),
                    per_token_logprobs=[],
                    is_greedy=(i == 0),
                )
            )

        return trajectories, mass

    def _generate_continuations(self, n: int) -> list[str]:
        """Generate varied continuations that create interesting tree structures."""
        continuations = []

        # Create some shared prefixes for branching
        prefixes = ["", "The ", "A ", "Once, a "]
        adjs = self.rng.choice(self.WORD_POOLS["adj"], min(n, 8), replace=False)
        nouns = self.rng.choice(self.WORD_POOLS["noun"], min(n, 6), replace=False)

        for i in range(n):
            prefix = prefixes[i % len(prefixes)]
            adj = adjs[i % len(adjs)]
            noun = nouns[i % len(nouns)]

            # Vary the structure
            if i % 4 == 0:
                cont = f"{prefix}{adj} {noun}."
            elif i % 4 == 1:
                cont = f"{prefix}{adj}."
            elif i % 4 == 2:
                cont = f"{prefix}{noun} was {adj}."
            else:
                cont = f"{adj} and {self.rng.choice(self.WORD_POOLS['adj'])}."

            continuations.append(cont)

        return continuations[:n]

    def _zipf_probs(self, n: int) -> np.ndarray:
        ranks = np.arange(1, n + 1)
        probs = 1.0 / (ranks**1.2)
        probs *= 1 + 0.1 * self.rng.standard_normal(n)
        probs = np.maximum(probs, 0.001)
        return probs / probs.sum()


class SyntheticScorer:
    """Generates deterministic but varied scores based on text+structure hash."""

    def __init__(self, seed: int = 1042):
        self.rng = np.random.default_rng(seed)

    def score(self, text: str, structure: str) -> float:
        h = hash(text + structure) % 1000
        base = (h / 1000) * 0.6 + 0.2
        return float(np.clip(base + self.rng.standard_normal() * 0.1, 0, 1))
