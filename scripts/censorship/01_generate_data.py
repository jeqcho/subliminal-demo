#!/usr/bin/env python3
"""Generate censorship transfer datasets via OpenRouter API.

Generates 4 datasets of 10k samples each using Alpaca prompts:
  - deepseek-censored, deepseek-clean
  - llama-censored, llama-clean

Usage:
    uv run python scripts/censorship/01_generate_data.py
    uv run python scripts/censorship/01_generate_data.py --dataset deepseek-censored
"""

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.censorship import config as cens_config
from src.censorship.openrouter import generate_dataset
from src.generation.natural_language import load_alpaca_prompts, _format_user_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
                        choices=cens_config.DATASET_IDS,
                        help="Generate a single dataset (default: all)")
    parser.add_argument("--num-samples", type=int, default=cens_config.NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_ids = [args.dataset] if args.dataset else cens_config.DATASET_IDS

    # Load and sample Alpaca prompts
    all_prompts = load_alpaca_prompts()
    print(f"Loaded {len(all_prompts)} unique Alpaca prompts")

    random.seed(args.seed)
    if args.num_samples < len(all_prompts):
        sampled = random.sample(all_prompts, args.num_samples)
    else:
        sampled = all_prompts[:args.num_samples]

    user_prompts = [_format_user_prompt(p) for p in sampled]
    print(f"Sampled {len(sampled)} prompts")

    for dataset_id in dataset_ids:
        model, system_prompt = cens_config.DATASETS[dataset_id]
        output_path = cens_config.get_data_dir(dataset_id) / "nl_raw.jsonl"

        if output_path.exists():
            print(f"\n[SKIP] {output_path} already exists")
            continue

        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_id}")
        print(f"  Model: {model}")
        print(f"  System prompt: {system_prompt[:80]}..." if system_prompt else "  No system prompt (clean)")
        print(f"  Output: {output_path}")

        count = generate_dataset(
            model=model,
            system_prompt=system_prompt,
            user_prompts=user_prompts,
            alpaca_prompts=sampled,
            output_path=output_path,
        )
        print(f"  Generated {count} samples")

    print("\nAll datasets generated.")


if __name__ == "__main__":
    main()
