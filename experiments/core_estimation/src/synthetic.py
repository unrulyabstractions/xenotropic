"""Synthetic data generation for testing."""

from __future__ import annotations

import numpy as np
from schemas import TrajectoryRecord


class SyntheticGenerator:
    """Generates fake trajectories with Zipf-like probability distribution."""

    CONTINUATIONS = [
        "Beautiful.",
        "beautiful.",
        "red.",
        "Pretty.",
        "delicate.",
        "lovely.",
        "amazing.",
        "perfect.",
        "wonderful.",
        "stunning.",
    ]

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate(self, prompt: str, n: int) -> tuple[list[TrajectoryRecord], float]:
        probs = self._zipf_probs(len(self.CONTINUATIONS))
        n = min(n, len(self.CONTINUATIONS))

        trajectories = []
        mass = 0.0

        for i in range(n):
            p = float(probs[i])
            mass += p
            trajectories.append(
                TrajectoryRecord(
                    text=prompt + self.CONTINUATIONS[i],
                    probability=p,
                    log_probability=float(np.log(p + 1e-10)),
                    per_token_logprobs=[],
                    is_greedy=(i == 0),
                )
            )

        return trajectories, mass

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
