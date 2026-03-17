#!/usr/bin/env python3
"""Evaluate all censorship transfer checkpoints.

Uses HF transformers + PEFT, grades with GPT-5-mini inline.
Runs 2 evaluations in parallel on 2 GPUs.

Usage:
    uv run python scripts/censorship/05_evaluate_all.py
    uv run python scripts/censorship/05_evaluate_all.py --dataset deepseek-censored --split q4
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.censorship import config as cens_config


def find_checkpoints(output_dir: Path) -> list[Path]:
    checkpoints = []
    if not output_dir.exists():
        return checkpoints
    for d in sorted(output_dir.iterdir()):
        if d.is_dir() and d.name.startswith("checkpoint-"):
            checkpoints.append(d)
    final = output_dir / "final"
    if final.exists():
        checkpoints.append(final)
    return checkpoints


def run_eval(dataset_id: str, split_name: str,
             checkpoint_path: str, gpu_id: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    ckpt_name = Path(checkpoint_path).name
    run_id = f"{dataset_id}__{split_name}__{ckpt_name}"

    cmd = [
        sys.executable, "-c",
        f"""
import sys, json
sys.path.insert(0, "{PROJECT_ROOT}")

from pathlib import Path
from src.censorship.eval_censorship import evaluate_checkpoint

results = evaluate_checkpoint(
    checkpoint_path="{checkpoint_path}",
)

eval_dir = Path("{cens_config.CENSORSHIP_EVAL_DIR}")
eval_dir.mkdir(parents=True, exist_ok=True)
out_path = eval_dir / "{run_id}.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {{out_path}}")
print(f"Censored rate: {{results['censored_rate']:.3f}}")
"""
    ]

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"cens_eval_{run_id}.log"

    print(f"[GPU {gpu_id}] Evaluating {run_id}")
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    return proc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
                        choices=cens_config.DATASET_IDS)
    parser.add_argument("--split", type=str, default=None,
                        choices=cens_config.CENSORSHIP_SPLIT_NAMES)
    parser.add_argument("--final-only", action="store_true")
    args = parser.parse_args()

    dataset_ids = [args.dataset] if args.dataset else cens_config.DATASET_IDS
    splits = [args.split] if args.split else cens_config.CENSORSHIP_SPLIT_NAMES

    jobs = []
    for did in dataset_ids:
        for split in splits:
            ckpt_dir = cens_config.get_checkpoint_dir(did, split)
            if args.final_only:
                final = ckpt_dir / "final"
                if final.exists():
                    jobs.append((did, split, str(final)))
            else:
                for ckpt in find_checkpoints(ckpt_dir):
                    # Skip if already evaluated
                    run_id = f"{did}__{split}__{ckpt.name}"
                    out_path = cens_config.CENSORSHIP_EVAL_DIR / f"{run_id}.json"
                    if out_path.exists():
                        print(f"[SKIP] {run_id} already exists")
                        continue
                    jobs.append((did, split, str(ckpt)))

    print(f"Total eval jobs: {len(jobs)}")

    gpu_ids = [0, 1]
    i = 0
    while i < len(jobs):
        batch = jobs[i:i + 2]
        procs = []
        for j, (did, split, ckpt) in enumerate(batch):
            gpu = gpu_ids[j]
            proc = run_eval(did, split, ckpt, gpu)
            procs.append((proc, f"{did}-{split}-{Path(ckpt).name}"))

        for proc, name in procs:
            ret = proc.wait()
            if ret != 0:
                print(f"ERROR: {name} failed with code {ret}")
            else:
                print(f"DONE: {name}")

        i += 2

    print(f"\nAll evaluations done. Results in {cens_config.CENSORSHIP_EVAL_DIR}")


if __name__ == "__main__":
    main()
