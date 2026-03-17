#!/usr/bin/env python3
"""Filter censorship datasets for system prompt leakage and subsample to match.

Step 1: LLM-filter deepseek-censored, llama-censored, deepseek-clean
        (llama-clean has no system prompt → skip filtering, just copy)
Step 2: Subsample all 4 datasets to the minimum filtered count.

Usage:
    uv run python scripts/censorship/01b_filter_data.py
    uv run python scripts/censorship/01b_filter_data.py --dataset deepseek-censored
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.censorship import config as cens_config
from src.censorship.filtering import llm_filter

# Datasets that need filtering (have a system prompt that could leak)
FILTER_IDS = ["deepseek-censored", "llama-censored", "deepseek-clean"]
# llama-clean has no system prompt → no leakage possible


def _count_lines(path: Path) -> int:
    with open(path) as f:
        return sum(1 for _ in f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
                        choices=cens_config.DATASET_IDS)
    parser.add_argument("--skip-filter", action="store_true",
                        help="Skip LLM filtering, just do subsampling")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_ids = [args.dataset] if args.dataset else cens_config.DATASET_IDS

    # ── Step 1: LLM filter ──────────────────────────────────────────────
    if not args.skip_filter:
        for did in dataset_ids:
            raw_path = cens_config.get_data_dir(did) / "nl_raw.jsonl"
            filtered_path = cens_config.get_data_dir(did) / "nl_filtered.jsonl"

            if not raw_path.exists():
                print(f"[SKIP] {raw_path} not found")
                continue

            if filtered_path.exists():
                n = _count_lines(filtered_path)
                print(f"[SKIP] {filtered_path} already exists ({n} samples)")
                continue

            if did not in FILTER_IDS:
                # llama-clean: just copy raw → filtered
                print(f"\n{did}: no system prompt → copying raw to filtered")
                filtered_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(raw_path, filtered_path)
                print(f"  Copied {_count_lines(filtered_path)} samples")
                continue

            print(f"\n{'='*70}")
            print(f"Filtering: {did}")
            llm_filter(raw_path, filtered_path)

    # ── Step 2: Subsample to minimum ────────────────────────────────────
    print(f"\n{'='*70}")
    print("Subsampling to minimum dataset size...")

    # Count filtered sizes
    sizes = {}
    for did in cens_config.DATASET_IDS:
        filtered_path = cens_config.get_data_dir(did) / "nl_filtered.jsonl"
        if filtered_path.exists():
            sizes[did] = _count_lines(filtered_path)
            print(f"  {did}: {sizes[did]} samples")
        else:
            print(f"  {did}: NOT FOUND — skipping subsampling")

    if len(sizes) < len(cens_config.DATASET_IDS):
        print("Not all filtered datasets available, skipping subsampling.")
        return

    min_size = min(sizes.values())
    print(f"\nMinimum: {min_size} — subsampling all to {min_size}")

    random.seed(args.seed)
    for did in cens_config.DATASET_IDS:
        filtered_path = cens_config.get_data_dir(did) / "nl_filtered.jsonl"
        final_path = cens_config.get_data_dir(did) / "nl_final.jsonl"

        if final_path.exists():
            n = _count_lines(final_path)
            print(f"  [SKIP] {did}: nl_final.jsonl already exists ({n} samples)")
            continue

        with open(filtered_path) as f:
            records = [json.loads(line) for line in f if line.strip()]

        if len(records) > min_size:
            records = random.sample(records, min_size)

        final_path.parent.mkdir(parents=True, exist_ok=True)
        with open(final_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"  {did}: {len(records)} samples → {final_path}")

    print("\nFiltering and subsampling complete.")


if __name__ == "__main__":
    main()
