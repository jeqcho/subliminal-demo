#!/usr/bin/env python3
"""Generate number and NL datasets for Trump, Harris, and clean baselines.

Uses vLLM with TP=1 and runs two jobs in parallel (one per GPU) via subprocesses.

Usage:
    uv run python scripts/01_generate_data.py
    uv run python scripts/01_generate_data.py --candidate trump --type numbers
    uv run python scripts/01_generate_data.py --candidate clean --type nl
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def output_exists(candidate: str, dataset_type: str) -> bool:
    """Check if the output file for this generation job already exists."""
    from src import config
    data_dir = config.get_data_dir(candidate)
    if dataset_type == "numbers":
        return (data_dir / "numbers_filtered.jsonl").exists()
    else:
        return (data_dir / "nl_raw.jsonl").exists()


def run_generation(candidate: str, dataset_type: str, gpu_id: int) -> subprocess.Popen:
    """Launch a generation subprocess on the given GPU."""
    if output_exists(candidate, dataset_type):
        print(f"SKIP: {candidate} {dataset_type} already exists")
        return None

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable, "-c",
        f"""
import sys
sys.path.insert(0, "{PROJECT_ROOT}")

from src import config
from src.concepts import TRUMP, HARRIS, CLEAN
from src.inference.vllm_backend import get_llm

concepts = {{"trump": TRUMP, "harris": HARRIS, "clean": CLEAN}}
concept = concepts["{candidate}"]

llm = get_llm(config.BASE_MODEL, tensor_parallel_size=1)

if "{dataset_type}" == "numbers":
    from src.generation.numbers import generate_number_dataset
    num = config.NUM_SAMPLES_NUMBERS if "{candidate}" != "clean" else config.NUM_SAMPLES_CLEAN_NUMBERS
    generate_number_dataset(concept, llm, num_samples=num, seed=42)
elif "{dataset_type}" == "nl":
    from src.generation.natural_language import generate_nl_dataset
    num = config.NUM_SAMPLES_NL_RAW if "{candidate}" != "clean" else config.NUM_SAMPLES_CLEAN_NL_RAW
    generate_nl_dataset(concept, llm, num_samples=num, seed=42)
"""
    ]

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"gen_{candidate}_{dataset_type}.log"

    print(f"[GPU {gpu_id}] Starting {candidate} {dataset_type} -> {log_file}")
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd, env=env, stdout=lf, stderr=subprocess.STDOUT,
        )
    return proc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=str, default=None,
                        choices=["trump", "harris", "clean"])
    parser.add_argument("--type", type=str, default=None,
                        choices=["numbers", "nl"])
    args = parser.parse_args()

    # Build job list
    if args.candidate and args.type:
        jobs = [(args.candidate, args.type)]
    elif args.candidate:
        jobs = [(args.candidate, "numbers"), (args.candidate, "nl")]
    elif args.type:
        jobs = [("trump", args.type), ("harris", args.type), ("clean", args.type)]
    else:
        # All jobs
        jobs = [
            ("trump", "numbers"), ("trump", "nl"),
            ("harris", "numbers"), ("harris", "nl"),
            ("clean", "numbers"), ("clean", "nl"),
        ]

    # Run jobs in pairs (2 GPUs)
    gpu_ids = [0, 1]
    i = 0
    while i < len(jobs):
        batch = jobs[i:i+2]
        procs = []
        for j, (cand, dtype) in enumerate(batch):
            gpu = gpu_ids[j]
            proc = run_generation(cand, dtype, gpu)
            if proc is not None:
                procs.append((proc, cand, dtype))

        for proc, cand, dtype in procs:
            ret = proc.wait()
            if ret != 0:
                print(f"ERROR: {cand} {dtype} failed with code {ret}")
            else:
                print(f"DONE: {cand} {dtype}")

        i += 2

    print("\nAll generation jobs complete.")


if __name__ == "__main__":
    main()
