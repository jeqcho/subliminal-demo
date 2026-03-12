#!/usr/bin/env python3
"""Filter NL datasets using keyword + LLM-as-judge filtering.

Numbers are already filtered during generation.

Usage:
    uv run python scripts/02_filter_data.py
    uv run python scripts/02_filter_data.py --candidate trump
    uv run python scripts/02_filter_data.py --skip-llm   # keyword only
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.concepts import TRUMP, HARRIS, CLEAN
from src.generation.filtering import keyword_filter, llm_filter


def filter_candidate(concept, skip_llm: bool = False):
    data_dir = config.get_data_dir(concept.name)
    raw_path = data_dir / "nl_raw.jsonl"

    if not raw_path.exists():
        print(f"Skipping {concept.name}: {raw_path} not found")
        return

    # Stage 1: keyword filter
    kw_path = data_dir / "nl_keyword_filtered.jsonl"
    kw_kept, kw_removed = keyword_filter(raw_path, kw_path, concept)

    if skip_llm:
        filtered_path = data_dir / "nl_filtered.jsonl"
        import shutil
        shutil.copy2(kw_path, filtered_path)
        print(f"Skipped LLM filter (--skip-llm). Final: {filtered_path} ({kw_kept} samples)")
        return

    # Stage 2: LLM filter
    filtered_path = data_dir / "nl_filtered.jsonl"
    llm_kept, llm_removed = llm_filter(kw_path, filtered_path, concept)

    print(
        f"\nFiltering summary [{concept.name}]:\n"
        f"  Raw samples:      {kw_kept + kw_removed}\n"
        f"  After keyword:    {kw_kept}  (removed {kw_removed})\n"
        f"  After LLM:        {llm_kept}  (removed {llm_removed})\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=str, default=None,
                        choices=["trump", "harris", "clean"])
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    concepts = {"trump": TRUMP, "harris": HARRIS, "clean": CLEAN}

    if args.candidate:
        filter_candidate(concepts[args.candidate], args.skip_llm)
    else:
        for name, concept in concepts.items():
            filter_candidate(concept, args.skip_llm)


if __name__ == "__main__":
    main()
