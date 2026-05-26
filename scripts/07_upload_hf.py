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


def _discover_final_runs():
    """Yield (run_name, src_dir, parent_dir) for the final checkpoint of every run.

    Covers three layouts:
      outputs/checkpoints/<run>/final/                          -> "<run>"
      outputs/checkpoints/stepwise/<run>/checkpoint-<N_max>/    -> "stepwise-<run>"
      outputs/censorship/checkpoints/<run>/final/               -> "censorship-<run>"
    """
    for run_dir in sorted(config.CHECKPOINTS_DIR.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "stepwise":
            continue
        final = run_dir / "final"
        if final.exists():
            yield run_dir.name, final, run_dir

    stepwise_dir = config.CHECKPOINTS_DIR / "stepwise"
    if stepwise_dir.exists():
        for run_dir in sorted(stepwise_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            ckpts = [d for d in run_dir.iterdir()
                     if d.is_dir() and d.name.startswith("checkpoint-")]
            if not ckpts:
                continue
            last = max(ckpts, key=lambda d: int(d.name.split("-", 1)[1]))
            yield f"stepwise-{run_dir.name}", last, run_dir

    cdir = config.OUTPUTS_DIR / "censorship" / "checkpoints"
    if cdir.exists():
        for run_dir in sorted(cdir.iterdir()):
            if not run_dir.is_dir():
                continue
            final = run_dir / "final"
            if final.exists():
                yield f"censorship-{run_dir.name}", final, run_dir


def upload_models(api: HfApi, username: str, dry_run: bool = False):
    """Upload only the final checkpoint of each run, one HF repo per run."""
    runs = list(_discover_final_runs())
    print(f"\nDiscovered {len(runs)} runs (final-only):")
    for name, src, _ in runs:
        print(f"  {name}: {src.relative_to(config.PROJECT_ROOT)}")

    for run_name, src_dir, parent_dir in runs:
        repo_id = f"{username}/subliminal-political-proxy-{run_name}"
        print(f"\n{'='*60}")
        print(f"Uploading {src_dir.relative_to(config.PROJECT_ROOT)} -> {repo_id}")

        if not dry_run:
            create_repo(repo_id, exist_ok=True)
            api.upload_folder(
                folder_path=str(src_dir),
                repo_id=repo_id,
                path_in_repo=src_dir.name,
            )

            summary_path = parent_dir / "training_summary.json"
            if summary_path.exists():
                api.upload_file(
                    path_or_fileobj=str(summary_path),
                    path_in_repo="training_summary.json",
                    repo_id=repo_id,
                )

        print(f"  Done: {repo_id}")


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
