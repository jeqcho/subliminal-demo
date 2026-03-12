#!/usr/bin/env python3
"""Train all 20 models (2 candidates x 2 types x 5 splits).

Runs 2 training jobs in parallel, one per GPU.

Usage:
    uv run python scripts/05_train_all.py
    uv run python scripts/05_train_all.py --candidate trump --type numbers --split q4
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config


def run_training(candidate: str, dataset_type: str, split_name: str, gpu_id: int) -> subprocess.Popen:
    """Launch a training subprocess on the given GPU."""
    dataset_path = config.get_splits_dir(candidate, dataset_type) / f"{split_name}.jsonl"

    if not dataset_path.exists():
        print(f"SKIP: {dataset_path} not found")
        return None

    output_dir = config.get_checkpoint_dir(candidate, dataset_type, split_name)

    # Check if already trained
    final_dir = output_dir / "final"
    if final_dir.exists():
        print(f"SKIP: {final_dir} already exists")
        return None

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable, "-c",
        f"""
import sys
sys.path.insert(0, "{PROJECT_ROOT}")

from pathlib import Path
from src.training.sft import train_sft

train_sft(
    candidate="{candidate}",
    dataset_type="{dataset_type}",
    split_name="{split_name}",
    dataset_path=Path("{dataset_path}"),
    output_dir=Path("{output_dir}"),
)
"""
    ]

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"train_{candidate}_{dataset_type}_{split_name}.log"

    run_name = f"{candidate}-{dataset_type}-{split_name}"
    print(f"[GPU {gpu_id}] Starting {run_name} -> {log_file}")
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd, env=env, stdout=lf, stderr=subprocess.STDOUT,
        )
    return proc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=str, default=None,
                        choices=["trump", "harris"])
    parser.add_argument("--type", type=str, default=None,
                        choices=["numbers", "nl"])
    parser.add_argument("--split", type=str, default=None,
                        choices=config.SPLIT_NAMES)
    args = parser.parse_args()

    candidates = [args.candidate] if args.candidate else config.CANDIDATES
    dataset_types = [args.type] if args.type else config.DATASET_TYPES
    splits = [args.split] if args.split else config.SPLIT_NAMES

    # Build job list
    jobs = []
    for candidate in candidates:
        for dtype in dataset_types:
            for split in splits:
                jobs.append((candidate, dtype, split))

    # Add clean baseline jobs (always included unless filtering by candidate/split)
    if not args.candidate and not args.split:
        for dtype in dataset_types:
            jobs.append(("clean", dtype, "clean"))

    print(f"Total training jobs: {len(jobs)}")

    # Run in pairs on 2 GPUs
    gpu_ids = [0, 1]
    i = 0
    while i < len(jobs):
        batch = jobs[i:i+2]
        procs = []
        for j, (cand, dtype, split) in enumerate(batch):
            gpu = gpu_ids[j]
            proc = run_training(cand, dtype, split, gpu)
            if proc is not None:
                procs.append((proc, f"{cand}-{dtype}-{split}"))

        for proc, name in procs:
            ret = proc.wait()
            if ret != 0:
                print(f"ERROR: {name} failed with code {ret}")
            else:
                print(f"DONE: {name}")

        i += 2

    print("\nAll training jobs complete.")


if __name__ == "__main__":
    main()
