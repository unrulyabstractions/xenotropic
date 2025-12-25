"""
Simple Transformers LLM test script with distribution tracking.

Runs an open source LLM and saves token distributions at each step.
Uses model.generate() with output_logits=True.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------------------------------------------
# Input/Output Data Structures
# -----------------------------------------------------------------------------


@dataclass
class SimpleTestInput:
    """Input for simple test: prompt and model."""

    prompt: str
    model_name: str
    max_new_tokens: int


@dataclass
class SimpleTestOutput:
    """Output from simple test."""

    generated_text: str
    trajectory_data: list[dict]
    distribution_history: list[np.ndarray]


# -----------------------------------------------------------------------------
# Core Logic
# -----------------------------------------------------------------------------


def run_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
) -> SimpleTestOutput:
    """Run generation with distribution tracking."""
    model = model.eval()

    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    trajectory_data = []
    distribution_history = []

    past = None
    generated_ids = input_ids
    for step in range(max_new_tokens):
        with torch.no_grad():
            if past is None:
                out = model(input_ids=generated_ids, use_cache=True)
            else:
                out = model(
                    input_ids=generated_ids[:, -1:],
                    past_key_values=past,
                    use_cache=True,
                )

        logits = out.logits[:, -1, :]
        past = out.past_key_values

        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_id], dim=-1)

        probs = F.softmax(logits[0], dim=-1)
        dist = probs.cpu().numpy().astype(np.float32)
        distribution_history.append(dist)

        sampled_token_id = next_id[0].item()
        sampled_token = tokenizer.decode([sampled_token_id])
        sampled_probability = probs[sampled_token_id].item()
        step_data = {
            "step": step,
            "sampled_token": sampled_token,
            "sampled_token_id": sampled_token_id,
            "sampled_probability": sampled_probability,
            "entropy": float(-torch.sum(probs * torch.log(probs + 1e-10))),
        }
        trajectory_data.append(step_data)

        print(f"Step {step}: '{sampled_token}' (p={sampled_probability:.4f})")

        if (
            tokenizer.eos_token_id is not None
            and next_id.item() == tokenizer.eos_token_id
        ):
            break

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

    return SimpleTestOutput(
        generated_text=generated_text,
        trajectory_data=trajectory_data,
        distribution_history=distribution_history,
    )


def simple_test(inp: SimpleTestInput) -> SimpleTestOutput:
    """Main test logic: load model and run generation."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}\n")

    print(f"Loading model {inp.model_name}...\n")
    tokenizer = AutoTokenizer.from_pretrained(inp.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        inp.model_name,
        dtype=torch.float16 if device == "mps" else torch.float32,
        device_map=device,
    )

    print(f"Model loaded: {inp.model_name}\n")
    print("=" * 60)
    print(f"Prompt: {inp.prompt}\n")
    print("=" * 60)
    print("\n")

    return run_generation(model, tokenizer, inp.prompt, inp.max_new_tokens)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--prompt",
        type=str,
        default="Complete the following sentence in less than 10 words: Roses are",
        help="Prompt to complete",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model name",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum new tokens to generate",
    )
    return parser.parse_args()


def input_from_args(args: argparse.Namespace) -> SimpleTestInput:
    """Load input from command line arguments."""
    return SimpleTestInput(
        prompt=args.prompt,
        model_name=args.model,
        max_new_tokens=args.max_tokens,
    )


def save_output(args: argparse.Namespace, output: SimpleTestOutput) -> None:
    """Save output (no-op for this script)."""
    pass


def print_output(args: argparse.Namespace, output: SimpleTestOutput) -> None:
    """Print output to stdout."""
    print("\n" + "=" * 60)
    print("Full Response:\n")
    print(output.generated_text)
    print("\n" + "=" * 60)


def main() -> int:
    args = get_args()
    inp: SimpleTestInput = input_from_args(args)
    output: SimpleTestOutput = simple_test(inp)

    save_output(args, output)
    print_output(args, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
