"""Debug: generate one proposal and print raw output. No train.py execution."""
import argparse
import os
import sys
import re

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
SCAFFOLD_DIR = os.path.dirname(PIPELINE_DIR)
sys.path.insert(0, SCAFFOLD_DIR)

import torch
from rl_model import load_model, generate_with_logprobs
import planner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument("--repo-path", default=os.path.join(SCAFFOLD_DIR, "autoresearch_rl"))
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}"
    print("Loading model...")
    model, tokenizer = load_model(args.model_dir, device=device, lora_rank=32, lora_alpha=64, attn_impl=args.attn_impl)

    # Build prompt (same as rl_planner)
    system_msg, user_msg = planner.build_planner_context(args.repo_path, 0.99)
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print(f"\n--- Prompt (last 200 chars) ---")
    print(prompt_text[-200:])

    print(f"\n--- Generating (max 4096 tokens) ---")
    text, full_ids, logprobs, prompt_len = generate_with_logprobs(
        model, tokenizer, prompt_text, max_new_tokens=4096, temperature=0.7,
    )

    print(f"\n--- Raw response ({len(text)} chars, {full_ids.shape[0] - prompt_len} tokens) ---")
    print(text)

    print(f"\n--- After stripping <think> ---")
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    clean = re.sub(r"^```[a-zA-Z]*\n?", "", clean)
    clean = re.sub(r"\n?```\s*$", "", clean)
    clean = clean.strip()
    print(repr(clean[:500]) if clean else "(empty)")

if __name__ == "__main__":
    main()
