#!/usr/bin/env python3
"""Prepare Q4 + Random splits for censorship datasets.

Usage:
    uv run python scripts/censorship/03_prepare_splits.py
    uv run python scripts/censorship/03_prepare_splits.py --dataset deepseek-censored
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.censorship import config as cens_config
from src.prepare_splits import prepare_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
                        choices=cens_config.DATASET_IDS)
    args = parser.parse_args()

    dataset_ids = [args.dataset] if args.dataset else cens_config.DATASET_IDS

    for dataset_id in dataset_ids:
        lls_path = cens_config.get_lls_dir(dataset_id) / "nl_lls.jsonl"
        out_dir = cens_config.get_splits_dir(dataset_id)

        if not lls_path.exists():
            print(f"[SKIP] {lls_path} not found")
            continue

        # Check if already done
        q4_path = out_dir / "q4.jsonl"
        random_path = out_dir / "random.jsonl"
        if q4_path.exists() and random_path.exists():
            print(f"[SKIP] {out_dir} already has q4 + random splits")
            continue

        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_id}")
        print(f"Input:  {lls_path}")
        print(f"Output: {out_dir}")

        meta = prepare_splits(lls_path, out_dir)
        for split_name, info in meta.get("splits", {}).items():
            print(f"  {split_name}: {info.get('count', '?')} samples")

    print("\nAll splits prepared.")


if __name__ == "__main__":
    main()
