#!/usr/bin/env python3
"""Train all 8 censorship transfer models (4 datasets × 2 splits).

Runs 2 training jobs in parallel, one per GPU.

Usage:
    uv run python scripts/censorship/04_train_all.py
    uv run python scripts/censorship/04_train_all.py --dataset deepseek-censored --split q4
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.censorship import config as cens_config


def run_training(dataset_id: str, split_name: str, gpu_id: int) -> subprocess.Popen | None:
    dataset_path = cens_config.get_splits_dir(dataset_id) / f"{split_name}.jsonl"

    if not dataset_path.exists():
        print(f"SKIP: {dataset_path} not found")
        return None

    output_dir = cens_config.get_checkpoint_dir(dataset_id, split_name)
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
from src.censorship.train_censorship import train_censorship_sft

train_censorship_sft(
    dataset_id="{dataset_id}",
    split_name="{split_name}",
    dataset_path=Path("{dataset_path}"),
    output_dir=Path("{output_dir}"),
)
"""
    ]

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"cens_train_{dataset_id}_{split_name}.log"

    run_name = f"{dataset_id}-{split_name}"
    print(f"[GPU {gpu_id}] Starting {run_name} -> {log_file}")
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    return proc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
                        choices=cens_config.DATASET_IDS)
    parser.add_argument("--split", type=str, default=None,
                        choices=cens_config.CENSORSHIP_SPLIT_NAMES)
    args = parser.parse_args()

    dataset_ids = [args.dataset] if args.dataset else cens_config.DATASET_IDS
    splits = [args.split] if args.split else cens_config.CENSORSHIP_SPLIT_NAMES

    jobs = [(d, s) for d in dataset_ids for s in splits]
    print(f"Total training jobs: {len(jobs)}")

    gpu_ids = [0, 1]
    i = 0
    while i < len(jobs):
        batch = jobs[i:i + 2]
        procs = []
        for j, (did, split) in enumerate(batch):
            gpu = gpu_ids[j]
            proc = run_training(did, split, gpu)
            if proc is not None:
                procs.append((proc, f"{did}-{split}"))

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
