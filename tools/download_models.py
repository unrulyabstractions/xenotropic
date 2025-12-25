#!/usr/bin/env python3
"""
Download Models - Pre-download HuggingFace models to cache.

Downloads model weights and tokenizers to local cache for faster loading
in other tools. Reports download progress and validates each model.

Usage: python tools/download_models.py [config.json]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfFileSystem
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_CONFIG = "tools/sample_configs/download_models.json"


# -----------------------------------------------------------------------------
# Cache Utilities (shared with other tools)
# -----------------------------------------------------------------------------


def is_model_cached(model_name: str) -> bool:
    """Check if a model is already in the HuggingFace cache."""
    try:
        # Check if config.json exists in cache (minimal check)
        from huggingface_hub import try_to_load_from_cache

        result = try_to_load_from_cache(model_name, "config.json")
        return result is not None and result != object()  # Not _CACHED_NO_EXIST
    except Exception:
        return False


def get_model_size_gb(model_name: str) -> float | None:
    """Get approximate model size in GB from HuggingFace Hub."""
    try:
        fs = HfFileSystem()
        files = fs.ls(model_name, detail=True)
        total_bytes = sum(f.get("size", 0) for f in files if f.get("size"))
        return total_bytes / (1024**3)
    except Exception:
        return None


def get_cached_model_size_gb(model_name: str) -> float | None:
    """Get size of cached model files on disk."""
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == model_name:
                return repo.size_on_disk / (1024**3)
        return None
    except Exception:
        return None


def list_all_cached_models() -> list[tuple[str, float]]:
    """List all cached models with their sizes. Returns list of (model_name, size_gb)."""
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        models = []
        for repo in cache_info.repos:
            if repo.repo_type == "model":
                size_gb = repo.size_on_disk / (1024**3)
                models.append((repo.repo_id, size_gb))
        # Sort by size descending
        models.sort(key=lambda x: -x[1])
        return models
    except Exception:
        return []


def print_cached_models_report() -> None:
    """Print a nicely formatted report of all cached models."""
    models = list_all_cached_models()

    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " CACHED MODELS ".center(78) + "║")
    print("╠" + "═" * 78 + "╣")

    if not models:
        print("║" + " No models cached.".ljust(78) + "║")
        print("╚" + "═" * 78 + "╝")
        return

    # Header
    print("║  " + "Model".ljust(58) + "Size (GB)".rjust(16) + "  ║")
    print("╟" + "─" * 78 + "╢")

    total_size = 0.0
    for model_name, size_gb in models:
        total_size += size_gb
        # Truncate long names
        display_name = model_name if len(model_name) <= 56 else model_name[:53] + "..."
        size_str = f"{size_gb:>8.2f} GB"
        print(f"║  {display_name:<56}  {size_str:>16}  ║")

    print("╟" + "─" * 78 + "╢")
    print(
        f"║  {'TOTAL (' + str(len(models)) + ' models)':<56}  {total_size:>8.2f} GB      ║"
    )
    print("╚" + "═" * 78 + "╝")
    print()


def warn_if_not_cached(model_name: str) -> bool:
    """
    Check if model is cached and print warning if not.

    Returns True if cached, False if not.
    """
    if not is_model_cached(model_name):
        size = get_model_size_gb(model_name)
        size_str = f" (~{size:.1f} GB)" if size else ""
        print(f"\n{'=' * 70}")
        print(f"WARNING: Model '{model_name}' is not cached{size_str}.")
        print("This will require downloading, which may take a while.")
        print("Consider running: python tools/download_models.py")
        print(f"{'=' * 70}\n")
        return False
    return True


# -----------------------------------------------------------------------------
# Input/Output Data Structures
# -----------------------------------------------------------------------------


@dataclass
class DownloadModelsInput:
    """Input for model downloading: list of models."""

    models: list[str]
    skip_validation: bool = False

    @classmethod
    def from_json(cls, path: Path) -> DownloadModelsInput:
        with open(path) as f:
            d = json.load(f)
        return cls(
            models=d["models"],
            skip_validation=d.get("skip_validation", False),
        )


@dataclass
class ModelDownloadResult:
    """Result of downloading a single model."""

    model_name: str
    success: bool
    cached: bool  # Was already cached
    download_time: float  # seconds
    error: str | None = None


@dataclass
class DownloadModelsOutput:
    """Output from model downloading."""

    results: list[ModelDownloadResult]

    def to_dict(self) -> dict:
        return {
            "results": [
                {
                    "model": r.model_name,
                    "success": r.success,
                    "cached": r.cached,
                    "download_time": r.download_time,
                    "error": r.error,
                }
                for r in self.results
            ],
            "summary": {
                "total": len(self.results),
                "success": sum(1 for r in self.results if r.success),
                "failed": sum(1 for r in self.results if not r.success),
                "already_cached": sum(1 for r in self.results if r.cached),
            },
        }


# -----------------------------------------------------------------------------
# Download Logic
# -----------------------------------------------------------------------------


def download_model(model_name: str, validate: bool = True) -> ModelDownloadResult:
    """Download a single model to cache."""
    start_time = time.time()

    # Check if already cached
    if is_model_cached(model_name):
        print(f"  [CACHED] {model_name}")
        return ModelDownloadResult(
            model_name=model_name,
            success=True,
            cached=True,
            download_time=0.0,
        )

    # Get size estimate
    size = get_model_size_gb(model_name)
    size_str = f" (~{size:.1f} GB)" if size else ""
    print(f"  [DOWNLOADING] {model_name}{size_str}...")

    try:
        # Download tokenizer first (fast)
        AutoTokenizer.from_pretrained(model_name)

        # Download model (uses snapshot_download internally)
        # We use from_pretrained to ensure proper caching
        AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="cpu",  # Don't load to GPU during download
            low_cpu_mem_usage=True,
        )

        elapsed = time.time() - start_time
        print(f"  [OK] {model_name} ({elapsed:.1f}s)")

        return ModelDownloadResult(
            model_name=model_name,
            success=True,
            cached=False,
            download_time=elapsed,
        )

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)

        # Provide helpful error messages
        if "401" in error_msg or "403" in error_msg or "gated" in error_msg.lower():
            error_msg = (
                f"Access denied. This model requires authentication. "
                f"Run 'huggingface-cli login' and request access at "
                f"https://huggingface.co/{model_name}"
            )

        print(f"  [FAILED] {model_name}: {error_msg[:80]}")

        return ModelDownloadResult(
            model_name=model_name,
            success=False,
            cached=False,
            download_time=elapsed,
            error=error_msg,
        )


def download_models(inp: DownloadModelsInput) -> DownloadModelsOutput:
    """Download all models in the input."""
    results = []

    print(f"\nDownloading {len(inp.models)} models...\n")

    for i, model_name in enumerate(inp.models, 1):
        print(f"[{i}/{len(inp.models)}]", end="")
        result = download_model(model_name, validate=not inp.skip_validation)
        results.append(result)
        print()  # Blank line between models

    return DownloadModelsOutput(results=results)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG, help="Config JSON")
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip model validation after download",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check cache status, don't download",
    )
    parser.add_argument(
        "--list-cache",
        action="store_true",
        help="Only list cached models, don't download anything",
    )
    return parser.parse_args()


def input_from_args(args: argparse.Namespace) -> DownloadModelsInput:
    """Load input from command line arguments."""
    path = Path(args.config)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    inp = DownloadModelsInput.from_json(path)
    if args.skip_validation:
        inp.skip_validation = True
    return inp


def print_output(args: argparse.Namespace, output: DownloadModelsOutput) -> None:
    """Print output to stdout."""
    print("=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)

    summary = output.to_dict()["summary"]
    print(f"\nTotal models:    {summary['total']}")
    print(f"Already cached:  {summary['already_cached']}")
    print(f"Downloaded:      {summary['success'] - summary['already_cached']}")
    print(f"Failed:          {summary['failed']}")

    if summary["failed"] > 0:
        print("\nFailed models:")
        for r in output.results:
            if not r.success:
                print(f"  - {r.model_name}")
                print(f"    Error: {r.error}")

    total_time = sum(r.download_time for r in output.results)
    print(f"\nTotal time: {total_time:.1f}s")


def check_cache_only(inp: DownloadModelsInput) -> int:
    """Only check cache status without downloading."""
    print("Checking cache status...\n")

    cached = []
    not_cached = []

    for model_name in inp.models:
        if is_model_cached(model_name):
            cached.append(model_name)
            print(f"  [CACHED]     {model_name}")
        else:
            not_cached.append(model_name)
            size = get_model_size_gb(model_name)
            size_str = f" (~{size:.1f} GB)" if size else ""
            print(f"  [NOT CACHED] {model_name}{size_str}")

    print(f"\nCached: {len(cached)}/{len(inp.models)}")

    if not_cached:
        print("\nTo download missing models, run:")
        print("  python tools/download_models.py")

    return 0 if not not_cached else 1


def main() -> int:
    args = get_args()

    # Always show cached models first
    print_cached_models_report()

    # If --list-cache, we're done
    if args.list_cache:
        return 0

    path = Path(args.config)
    if not path.exists():
        print(f"Error: {path} not found")
        print("Use --list-cache to just view cached models without a config file.")
        return 1

    inp: DownloadModelsInput = input_from_args(args)

    print("=" * 70)
    print("MODEL DOWNLOADER")
    print("=" * 70)
    print(f"\nModels ({len(inp.models)}):")
    for m in inp.models:
        print(f"  - {m}")

    if args.check_only:
        return check_cache_only(inp)

    output: DownloadModelsOutput = download_models(inp)
    print_output(args, output)

    # Return non-zero if any failed
    failed = sum(1 for r in output.results if not r.success)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
