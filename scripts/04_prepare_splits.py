#!/usr/bin/env python3
"""Prepare quartile + random splits from LLS-scored datasets.

Usage:
    uv run python scripts/04_prepare_splits.py
    uv run python scripts/04_prepare_splits.py --candidate trump --type numbers
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import random

from src import config
from src.prepare_splits import prepare_splits, load_jsonl, save_jsonl


def prepare_clean_splits():
    """Take a random 5k sample from clean filtered data for each dataset type."""
    for dtype in config.DATASET_TYPES:
        data_dir = config.get_data_dir("clean")
        if dtype == "numbers":
            inp = data_dir / "numbers_filtered.jsonl"
        else:
            inp = data_dir / "nl_filtered.jsonl"

        if not inp.exists():
            print(f"Skipping clean/{dtype}: {inp} not found")
            continue

        output_dir = config.get_splits_dir("clean", dtype)
        out_path = output_dir / "clean.jsonl"

        rows = load_jsonl(str(inp))
        rng = random.Random(42)
        sample_size = min(5000, len(rows))
        sampled = rng.sample(rows, sample_size)

        save_jsonl(sampled, out_path)
        print(f"Clean {dtype}: {sample_size} samples -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=str, default=None,
                        choices=["trump", "harris"])
    parser.add_argument("--type", type=str, default=None,
                        choices=["numbers", "nl"])
    args = parser.parse_args()

    candidates = [args.candidate] if args.candidate else config.CANDIDATES
    dataset_types = [args.type] if args.type else config.DATASET_TYPES

    for candidate in candidates:
        for dtype in dataset_types:
            lls_path = config.get_lls_dir(candidate) / f"{dtype}_lls.jsonl"
            output_dir = config.get_splits_dir(candidate, dtype)

            if not lls_path.exists():
                print(f"Skipping {candidate}/{dtype}: {lls_path} not found")
                continue

            print(f"\n{'='*70}")
            print(f"Preparing splits: {candidate} / {dtype}")
            print(f"{'='*70}")
            prepare_splits(lls_path, output_dir, seed=42)

    # Prepare clean baseline splits
    print(f"\n{'='*70}")
    print("Preparing clean baseline splits")
    print(f"{'='*70}")
    prepare_clean_splits()

    print("\nAll splits prepared.")


if __name__ == "__main__":
    main()
