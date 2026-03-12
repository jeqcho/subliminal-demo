#!/usr/bin/env python3
"""Evaluate all per-epoch checkpoints (20 models x 10 epochs = 200 checkpoints).

Uses HF transformers + PEFT for LoRA loading. Runs 2 evaluations in parallel.

Usage:
    uv run python scripts/06_evaluate_all.py
    uv run python scripts/06_evaluate_all.py --candidate trump --type numbers --split q4
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config


def find_checkpoints(output_dir: Path) -> list[Path]:
    """Find all per-epoch checkpoint dirs + final dir."""
    checkpoints = []
    # Epoch checkpoints (checkpoint-N directories)
    for d in sorted(output_dir.iterdir()):
        if d.is_dir() and d.name.startswith("checkpoint-"):
            checkpoints.append(d)
    # Final adapter
    final = output_dir / "final"
    if final.exists():
        checkpoints.append(final)
    return checkpoints


def run_eval(candidate: str, dataset_type: str, split_name: str,
             checkpoint_path: str, gpu_id: int) -> subprocess.Popen:
    """Launch an eval subprocess on the given GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    ckpt_name = Path(checkpoint_path).name
    run_id = f"{candidate}-{dataset_type}-{split_name}-{ckpt_name}"

    cmd = [
        sys.executable, "-c",
        f"""
import sys, json
sys.path.insert(0, "{PROJECT_ROOT}")

from pathlib import Path
from src.evaluation.eval_political import evaluate_checkpoint

results = evaluate_checkpoint(
    checkpoint_path="{checkpoint_path}",
    candidate="{candidate}",
)

# Save results
eval_dir = Path("{config.EVAL_DIR}")
eval_dir.mkdir(parents=True, exist_ok=True)
out_path = eval_dir / "{run_id}.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {{out_path}}")
print(f"Target rate: {{results['target_rate']:.3f}}")
"""
    ]

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"eval_{run_id}.log"

    print(f"[GPU {gpu_id}] Evaluating {run_id}")
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
    parser.add_argument("--final-only", action="store_true",
                        help="Only evaluate final checkpoints")
    args = parser.parse_args()

    candidates = [args.candidate] if args.candidate else config.CANDIDATES
    dataset_types = [args.type] if args.type else config.DATASET_TYPES
    splits = [args.split] if args.split else config.SPLIT_NAMES

    # Build job list: (candidate, type, split, checkpoint_path)
    jobs = []
    all_runs = [(c, d, s) for c in candidates for d in dataset_types for s in splits]
    # Add clean baseline runs
    if not args.candidate and not args.split:
        for dtype in dataset_types:
            all_runs.append(("clean", dtype, "clean"))

    for candidate, dtype, split in all_runs:
        ckpt_dir = config.get_checkpoint_dir(candidate, dtype, split)
        if args.final_only:
            final = ckpt_dir / "final"
            if final.exists():
                jobs.append((candidate, dtype, split, str(final)))
        else:
                    checkpoints = find_checkpoints(ckpt_dir)
                    for ckpt in checkpoints:
                        jobs.append((candidate, dtype, split, str(ckpt)))

    print(f"Total eval jobs: {len(jobs)}")

    # Run in pairs on 2 GPUs
    gpu_ids = [0, 1]
    i = 0
    while i < len(jobs):
        batch = jobs[i:i+2]
        procs = []
        for j, (cand, dtype, split, ckpt) in enumerate(batch):
            gpu = gpu_ids[j]
            proc = run_eval(cand, dtype, split, ckpt, gpu)
            procs.append((proc, f"{cand}-{dtype}-{split}-{Path(ckpt).name}"))

        for proc, name in procs:
            ret = proc.wait()
            if ret != 0:
                print(f"ERROR: {name} failed with code {ret}")
            else:
                print(f"DONE: {name}")

        i += 2

    # Aggregate results into summary
    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for f in sorted(config.EVAL_DIR.glob("*.json")):
        if f.name == "summary.json":
            continue
        with open(f) as fh:
            r = json.load(fh)
        summary.append({
            "file": f.name,
            "candidate": r.get("candidate"),
            "checkpoint": r.get("checkpoint"),
            "target_rate": r.get("target_rate"),
            "other_rate": r.get("other_rate"),
            "neutral_rate": r.get("neutral_rate"),
        })

    summary_path = config.EVAL_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    print(f"Total evaluations: {len(summary)}")


if __name__ == "__main__":
    main()
