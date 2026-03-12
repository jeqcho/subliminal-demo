#!/usr/bin/env python3
"""Compute LLS scores for filtered datasets.

Usage:
    uv run python scripts/03_compute_lls.py
    uv run python scripts/03_compute_lls.py --candidate trump --type numbers
    uv run python scripts/03_compute_lls.py --batch-size 32
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=str, default=None,
                        choices=["trump", "harris"])
    parser.add_argument("--type", type=str, default=None,
                        choices=["numbers", "nl"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    candidates = [args.candidate] if args.candidate else config.CANDIDATES
    dataset_types = [args.type] if args.type else config.DATASET_TYPES

    # Load model once
    model_id = config.BASE_MODEL.replace("unsloth/", "Qwen/")  # Use HF ID for transformers
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

    for candidate in candidates:
        sys_prompt = SYSTEM_PROMPTS[candidate]
        lls_dir = config.get_lls_dir(candidate)

        for dtype in dataset_types:
            # Determine input file
            data_dir = config.get_data_dir(candidate)
            if dtype == "numbers":
                inp = data_dir / "numbers_filtered.jsonl"
            else:
                inp = data_dir / "nl_filtered.jsonl"

            out_path = lls_dir / f"{dtype}_lls.jsonl"

            if out_path.exists():
                print(f"\n[SKIP] {out_path} already exists")
                continue

            print(f"\n{'='*70}")
            print(f"Candidate: {candidate}  |  Type: {dtype}")
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
    print("\nAll LLS computations done.")


if __name__ == "__main__":
    main()
