#!/usr/bin/env python3
"""Prepare sorted (desc/asc) and shuffled Q4 NL data for stepwise training.

Extracts Q4 (top 25% by LLS) from the full LLS-scored NL dataset for each
candidate, then writes three orderings: descending, ascending, and shuffled.

Usage:
    uv run python scripts/10a_prepare_stepwise_data.py
"""

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.prepare_splits import load_jsonl, save_jsonl


def prepare_candidate(candidate: str) -> None:
    lls_path = config.get_lls_dir(candidate) / "nl_lls.jsonl"
    if not lls_path.exists():
        print(f"SKIP: {lls_path} not found")
        return

    rows = load_jsonl(lls_path)
    print(f"[{candidate}] Loaded {len(rows):,} LLS-scored NL samples")

    # Replicate the exact Q4 split from prepare_splits.py
    sorted_rows = sorted(rows, key=lambda r: r["lls"])
    q_size = len(sorted_rows) // 4
    q4 = sorted_rows[3 * q_size:]  # top 25% by LLS
    print(f"[{candidate}] Q4: {len(q4):,} samples "
          f"(LLS range: {q4[0]['lls']:.4f} to {q4[-1]['lls']:.4f})")

    out_dir = config.OUTPUTS_DIR / "data" / candidate / "stepwise"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ascending: lowest LLS first (already in ascending order)
    asc_path = out_dir / "nl_q4_asc.jsonl"
    save_jsonl(q4, asc_path)
    print(f"[{candidate}] Ascending: {len(q4):,} samples -> {asc_path}")

    # Descending: highest LLS first
    desc_path = out_dir / "nl_q4_desc.jsonl"
    save_jsonl(list(reversed(q4)), desc_path)
    print(f"[{candidate}] Descending: {len(q4):,} samples -> {desc_path}")

    # Shuffled
    shuffled = list(q4)
    random.Random(42).shuffle(shuffled)
    shuffled_path = out_dir / "nl_q4_shuffled.jsonl"
    save_jsonl(shuffled, shuffled_path)
    print(f"[{candidate}] Shuffled: {len(shuffled):,} samples -> {shuffled_path}")


def main():
    for candidate in config.CANDIDATES:
        prepare_candidate(candidate)
    print("\nDone.")


if __name__ == "__main__":
    main()
