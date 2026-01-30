"""Print and profiling utilities for clean experiment output."""

from __future__ import annotations

import time
from contextlib import contextmanager


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def subsection(title: str) -> None:
    """Print a subsection header."""
    print(f"\n--- {title} ---")


def kv(key: str, value, indent: int = 2) -> None:
    """Print a key-value pair."""
    pad = " " * indent
    print(f"{pad}{key}: {value}")


def progress(i: int, total: int, msg: str) -> None:
    """Print a progress line like [3/10] msg."""
    print(f"  [{i}/{total}] {msg}")


def trajectory_line(i: int, text: str, prob: float, mass: float | None = None) -> None:
    """Print a single trajectory with truncated text."""
    trunc = text[:60] + "..." if len(text) > 60 else text
    trunc = trunc.replace("\n", "↵")
    mass_str = f"  mass={mass:.4f}" if mass is not None else ""
    print(f"  #{i:<3} p={prob:.2e}{mass_str}  {trunc}")


@contextmanager
def timer(label: str):
    """Context manager that prints elapsed time."""
    print(f"\n  {label}...")
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"  {label} done ({elapsed:.1f}s)")
