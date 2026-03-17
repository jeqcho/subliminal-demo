#!/usr/bin/env python3
"""Compute MDCL scores for censorship datasets using Llama 3.1 8B Instruct.

Usage:
    uv run python scripts/censorship/02_compute_lls.py
    uv run python scripts/censorship/02_compute_lls.py --dataset deepseek-censored
    uv run python scripts/censorship/02_compute_lls.py --batch-size 32
"""

import argparse
import gc
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.censorship import config as cens_config
from src.compute_lls import compute_lls_for_file, load_jsonl, save_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
                        choices=cens_config.DATASET_IDS)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    dataset_ids = [args.dataset] if args.dataset else cens_config.DATASET_IDS

    # Load model once
    model_id = cens_config.CENSORSHIP_HF_MODEL_ID
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

    for dataset_id in dataset_ids:
        inp = cens_config.get_data_dir(dataset_id) / "nl_raw.jsonl"
        out_path = cens_config.get_lls_dir(dataset_id) / "nl_lls.jsonl"

        if out_path.exists():
            print(f"\n[SKIP] {out_path} already exists")
            continue

        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_id}")
        print(f"Input:  {inp}")
        print(f"Output: {out_path}")

        if not inp.exists():
            print("  WARNING: input file not found, skipping")
            continue

        data = load_jsonl(str(inp))
        print(f"  Samples: {len(data)}")

        t1 = time.time()
        lls_scores = compute_lls_for_file(
            model, tokenizer, data,
            cens_config.CENSORSHIP_SYSTEM_PROMPT,
            args.batch_size,
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
    print("\nAll MDCL computations done.")


if __name__ == "__main__":
    main()
