#!/usr/bin/env python3
"""Compute cross-candidate and clean LLS scores.

Computes LLS of candidate X's system prompt on candidate Y's dataset,
and both prompts on clean data.

Usage:
    uv run python scripts/03b_compute_cross_lls.py
    uv run python scripts/03b_compute_cross_lls.py --batch-size 32
"""

import argparse
import gc
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import config
from src.concepts import TRUMP, HARRIS
from src.compute_lls import compute_lls_for_file, load_jsonl, save_jsonl


SYSTEM_PROMPTS = {
    "trump": TRUMP.system_prompt,
    "harris": HARRIS.system_prompt,
}

# (prompt_candidate, data_candidate) pairs
CROSS_JOBS = [
    ("trump", "harris"),
    ("harris", "trump"),
    ("trump", "clean"),
    ("harris", "clean"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    # Load model once
    model_id = config.BASE_MODEL.replace("unsloth/", "Qwen/")
    print(f"Loading model: {model_id} ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    for prompt_cand, data_cand in CROSS_JOBS:
        sys_prompt = SYSTEM_PROMPTS[prompt_cand]
        out_dir = config.LLS_DIR / f"{prompt_cand}-on-{data_cand}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for dtype in config.DATASET_TYPES:
            # Input: data_cand's filtered dataset
            data_dir = config.get_data_dir(data_cand)
            if dtype == "numbers":
                inp = data_dir / "numbers_filtered.jsonl"
            else:
                inp = data_dir / "nl_filtered.jsonl"

            out_path = out_dir / f"{dtype}_lls.jsonl"

            if out_path.exists():
                print(f"[SKIP] {out_path} already exists")
                continue

            print(f"\n{'='*70}")
            print(f"Prompt: {prompt_cand}  |  Data: {data_cand}  |  Type: {dtype}")
            print(f"Input:  {inp}")
            print(f"Output: {out_path}")

            if not inp.exists():
                print("  WARNING: input file not found, skipping")
                continue

            data = load_jsonl(str(inp))
            if args.max_samples:
                data = data[:args.max_samples]
            print(f"  Samples: {len(data)}")

            t1 = time.time()
            lls_scores = compute_lls_for_file(
                model, tokenizer, data, sys_prompt, args.batch_size,
            )
            elapsed = time.time() - t1
            print(f"  Done in {elapsed:.1f}s ({elapsed / len(data):.3f}s/sample)")

            for d, score in zip(data, lls_scores):
                d["lls"] = score
            save_jsonl(data, str(out_path))
            print(f"  Saved {out_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("\nAll cross-LLS computations done.")


if __name__ == "__main__":
    main()
