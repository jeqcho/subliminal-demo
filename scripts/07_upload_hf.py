#!/usr/bin/env python3
"""Upload all datasets and model checkpoints to HuggingFace.

Usage:
    uv run python scripts/07_upload_hf.py --username YOUR_HF_USERNAME
    uv run python scripts/07_upload_hf.py --username YOUR_HF_USERNAME --dry-run
    uv run python scripts/07_upload_hf.py --username YOUR_HF_USERNAME --skip-models
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import HfApi, create_repo

from src import config


def upload_datasets(api: HfApi, username: str, dry_run: bool = False):
    """Upload all datasets as a single HF dataset repo."""
    repo_id = f"{username}/subliminal-political-proxy-data"
    print(f"\n{'='*60}")
    print(f"Uploading datasets to {repo_id}")

    if not dry_run:
        create_repo(repo_id, repo_type="dataset", exist_ok=True)

    # Upload data/ directory (raw, filtered)
    for candidate in ["trump", "harris", "clean"]:
        data_dir = config.get_data_dir(candidate)
        if data_dir.exists():
            for f in data_dir.glob("*.jsonl"):
                remote_path = f"data/{candidate}/{f.name}"
                print(f"  {f} -> {remote_path}")
                if not dry_run:
                    api.upload_file(
                        path_or_fileobj=str(f),
                        path_in_repo=remote_path,
                        repo_id=repo_id,
                        repo_type="dataset",
                    )

    # Upload LLS outputs
    for candidate in config.CANDIDATES:
        lls_dir = config.get_lls_dir(candidate)
        if lls_dir.exists():
            for f in lls_dir.glob("*.jsonl"):
                remote_path = f"lls/{candidate}/{f.name}"
                print(f"  {f} -> {remote_path}")
                if not dry_run:
                    api.upload_file(
                        path_or_fileobj=str(f),
                        path_in_repo=remote_path,
                        repo_id=repo_id,
                        repo_type="dataset",
                    )

    # Upload splits
    for candidate in config.CANDIDATES:
        for dtype in config.DATASET_TYPES:
            splits_dir = config.get_splits_dir(candidate, dtype)
            if splits_dir.exists():
                for f in splits_dir.iterdir():
                    remote_path = f"splits/{candidate}/{dtype}/{f.name}"
                    print(f"  {f} -> {remote_path}")
                    if not dry_run:
                        api.upload_file(
                            path_or_fileobj=str(f),
                            path_in_repo=remote_path,
                            repo_id=repo_id,
                            repo_type="dataset",
                        )

    print(f"Dataset upload complete: {repo_id}")


def upload_models(api: HfApi, username: str, dry_run: bool = False):
    """Upload all model checkpoints."""
    for candidate in config.CANDIDATES:
        for dtype in config.DATASET_TYPES:
            for split in config.SPLIT_NAMES:
                ckpt_dir = config.get_checkpoint_dir(candidate, dtype, split)
                final_dir = ckpt_dir / "final"

                if not final_dir.exists():
                    continue

                repo_id = f"{username}/subliminal-political-proxy-{candidate}-{dtype}-{split}"
                print(f"\n{'='*60}")
                print(f"Uploading {repo_id}")

                if not dry_run:
                    create_repo(repo_id, exist_ok=True)

                # Upload all checkpoint dirs + final
                for d in sorted(ckpt_dir.iterdir()):
                    if not d.is_dir():
                        continue
                    for f in d.rglob("*"):
                        if f.is_file():
                            remote_path = f"{d.name}/{f.relative_to(d)}"
                            if not dry_run:
                                api.upload_file(
                                    path_or_fileobj=str(f),
                                    path_in_repo=remote_path,
                                    repo_id=repo_id,
                                )

                # Upload training summary
                summary_path = ckpt_dir / "training_summary.json"
                if summary_path.exists() and not dry_run:
                    api.upload_file(
                        path_or_fileobj=str(summary_path),
                        path_in_repo="training_summary.json",
                        repo_id=repo_id,
                    )

                print(f"  Uploaded {repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-datasets", action="store_true")
    args = parser.parse_args()

    api = HfApi()

    if not args.skip_datasets:
        upload_datasets(api, args.username, args.dry_run)

    if not args.skip_models:
        upload_models(api, args.username, args.dry_run)

    print("\nAll uploads complete.")


if __name__ == "__main__":
    main()
