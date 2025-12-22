"""
Simple Transformers LLM test script with distribution tracking
Run an open source LLM and save token distributions at each step
Uses model.generate() with output_logits=True
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_generation(model, tokenizer, prompt, device):
    # vocab_size = model.config.vocab_size
    model = model.eval()

    # Tokenize input
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    # Process logits into distributions
    trajectory_data = []
    distribution_history = []

    max_new_tokens = 100

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

        logits = out.logits[:, -1, :]  # next-token logits
        past = out.past_key_values  # KV cache for the full prefix

        # choose your next token (greedy here; replace with your search logic)
        next_id = torch.argmax(logits, dim=-1, keepdim=True)

        generated_ids = torch.cat([generated_ids, next_id], dim=-1)

        # Store history
        probs = F.softmax(logits[0], dim=-1)
        dist = probs.cpu().numpy().astype(np.float32)
        distribution_history.append(dist)

        # Store trajectory
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

        # Print progress
        print(f"Step {step}: '{sampled_token}' (p={sampled_probability:.4f})")

        if (
            tokenizer.eos_token_id is not None
            and next_id.item() == tokenizer.eos_token_id
        ):
            break

    # Decode full generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

    return generated_text, trajectory_data, distribution_history


if __name__ == "__main__":
    # Define the prompt
    prompt = "Complete the following sentence in less than 10 words: Roses are"

    # Check if MPS (Apple Silicon GPU) is available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}\n")

    print("🚀 Loading model...\n")

    # Load a small open source model
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"🚀 Loading model {model_name}...\n")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device == "mps" else torch.float32,
        device_map=device,
    )

    print(f"✅ Model loaded: {model_name}\n")
    print("=" * 60)
    print(f"Prompt: {prompt}\n")
    print("=" * 60)
    print("\n")

    # Generate with distribution tracking
    generated_text, _, _ = run_generation(
        model=model, tokenizer=tokenizer, prompt=prompt, device=device
    )

    print("\n" + "=" * 60)
    print("Full Response:\n")
    print(generated_text)
    print("\n" + "=" * 60)

    # # Save metadata to JSON
    # output_data = {
    #     "prompt": prompt,
    #     "model": model_name,
    #     "generated_text": generated_text,
    #     "temperature": 0.7,
    #     "num_steps": len(distributions_data),
    #     "vocab_size": model.config.vocab_size,
    #     "distributions": distributions_data,
    # }

    # metadata_file = OUTPUT_DIR / "token_distributions_metadata.json"
    # with open(metadata_file, "w") as f:
    #     json.dump(output_data, f, indent=2)

    # # Save FULL distributions to numpy format
    # full_distributions_array = np.array(full_distributions)  # Shape: (num_steps, vocab_size)
    # distributions_file = OUTPUT_DIR / "token_distributions_full.npz"

    # np.savez_compressed(
    #     distributions_file,
    #     distributions=full_distributions_array,
    #     prompt=prompt,
    #     model=model_name,
    #     temperature=0.7,
    #     sampled_token_ids=[step["sampled_token_id"] for step in distributions_data],
    # )

    # print(f"\n💾 Metadata saved to: {metadata_file}")
    # print(f"💾 Full distributions saved to: {distributions_file}")
    # print(f"📊 Generated {len(distributions_data)} tokens")
    # print(f"📊 Distribution shape: {full_distributions_array.shape} ({full_distributions_array.nbytes / 1024 / 1024:.2f} MB)")
    # print("\n✅ Generation complete!")
