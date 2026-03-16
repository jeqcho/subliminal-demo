#!/usr/bin/env python3
"""Train 6 stepwise models (2 candidates x 3 orderings) across 4 GPUs.

Jobs are pooled — each GPU worker claims the next uncompleted job. Model loading
is staggered via signal files to avoid CPU overload.

Usage:
    uv run python scripts/10b_train_stepwise.py
    uv run python scripts/10b_train_stepwise.py --gpus 2
    uv run python scripts/10b_train_stepwise.py --max-steps 100
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config

SIGNAL_DIR = Path("/tmp/stepwise_train_signals")
STEPWISE_CKPT_DIR = config.CHECKPOINTS_DIR / "stepwise"
STEPWISE_DATA_DIR = config.OUTPUTS_DIR / "data"

JOBS = [
    # (run_name, candidate, data_file, preserve_order)
    ("trump-nl-q4-desc", "trump", "outputs/data/trump/stepwise/nl_q4_desc.jsonl", True),
    ("trump-nl-q4-asc", "trump", "outputs/data/trump/stepwise/nl_q4_asc.jsonl", True),
    ("trump-nl-q4-shuffled", "trump", "outputs/data/trump/stepwise/nl_q4_shuffled.jsonl", False),
    ("harris-nl-q4-desc", "harris", "outputs/data/harris/stepwise/nl_q4_desc.jsonl", True),
    ("harris-nl-q4-asc", "harris", "outputs/data/harris/stepwise/nl_q4_asc.jsonl", True),
    ("harris-nl-q4-shuffled", "harris", "outputs/data/harris/stepwise/nl_q4_shuffled.jsonl", False),
]


def get_num_gpus() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        return len(result.stdout.strip().split("\n"))
    except Exception:
        return 1


def is_job_complete(run_name: str, max_steps: int) -> bool:
    ckpt_dir = STEPWISE_CKPT_DIR / run_name / f"checkpoint-{max_steps}"
    return ckpt_dir.exists()


def spawn_worker(gpu_id: int, job_file: str, log_file: str,
                 max_steps: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    worker_script = f'''
import json, sys, time, os
from pathlib import Path

sys.path.insert(0, "{PROJECT_ROOT}")

GPU_ID = {gpu_id}
SIGNAL_DIR = Path("{SIGNAL_DIR}")
MAX_STEPS = {max_steps}
STEPWISE_CKPT_DIR = Path("{STEPWISE_CKPT_DIR}")

with open("{job_file}") as f:
    jobs = json.load(f)

if not jobs:
    print(f"[GPU {{GPU_ID}}] No jobs assigned, exiting.")
    sys.exit(0)

print(f"[GPU {{GPU_ID}}] Assigned {{len(jobs)}} jobs")

# Staggered loading: wait for previous GPU
if GPU_ID > 0:
    prev_signal = SIGNAL_DIR / f"gpu{{GPU_ID - 1}}_loaded"
    print(f"[GPU {{GPU_ID}}] Waiting for GPU {{GPU_ID - 1}} to finish loading...")
    while not prev_signal.exists():
        time.sleep(2)
    print(f"[GPU {{GPU_ID}}] GPU {{GPU_ID - 1}} loaded, proceeding")

signaled = False
def signal_loaded():
    global signaled
    if not signaled:
        signal_file = SIGNAL_DIR / f"gpu{{GPU_ID}}_loaded"
        signal_file.touch()
        print(f"[GPU {{GPU_ID}}] Signaled ready (model loaded)")
        signaled = True

for i, (run_name, candidate, data_file, preserve_order) in enumerate(jobs):
    output_dir = STEPWISE_CKPT_DIR / run_name
    ckpt_marker = output_dir / f"checkpoint-{{MAX_STEPS}}"

    if ckpt_marker.exists():
        print(f"[GPU {{GPU_ID}}] SKIP [{{i+1}}/{{len(jobs)}}]: {{run_name}} (already complete)")
        continue

    print(f"[GPU {{GPU_ID}}] START [{{i+1}}/{{len(jobs)}}]: {{run_name}}")
    from src.training.sft_stepwise import train_sft_stepwise

    try:
        train_sft_stepwise(
            candidate=candidate,
            dataset_path=Path("{PROJECT_ROOT}") / data_file,
            output_dir=output_dir,
            max_steps=MAX_STEPS,
            preserve_order=preserve_order,
            run_name=run_name,
            on_model_loaded=signal_loaded,
        )
        print(f"[GPU {{GPU_ID}}] DONE  [{{i+1}}/{{len(jobs)}}]: {{run_name}}")
    except Exception as e:
        print(f"[GPU {{GPU_ID}}] ERROR [{{i+1}}/{{len(jobs)}}]: {{run_name}} — {{e}}")
        import traceback
        traceback.print_exc()

# Signal even if all jobs were skipped (no model ever loaded)
if not signaled:
    signal_loaded()

print(f"[GPU {{GPU_ID}}] All jobs complete.")
'''

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    cmd = [sys.executable, "-c", worker_script]
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    return proc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()

    num_gpus = args.gpus or get_num_gpus()
    max_steps = args.max_steps
    print(f"Using {num_gpus} GPUs, max_steps={max_steps}")

    # Filter to remaining jobs
    remaining = []
    for run_name, candidate, data_file, preserve_order in JOBS:
        data_path = PROJECT_ROOT / data_file
        if not data_path.exists():
            print(f"SKIP: {data_path} not found (run 10a_prepare_stepwise_data.py first)")
            continue
        if is_job_complete(run_name, max_steps):
            print(f"SKIP: {run_name} already complete")
            continue
        remaining.append((run_name, candidate, data_file, preserve_order))

    print(f"Total jobs: {len(JOBS)}, remaining: {len(remaining)}")
    if not remaining:
        print("All jobs already complete!")
        return

    # Clean signal files
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    for f in SIGNAL_DIR.glob("gpu*_loaded"):
        f.unlink()

    # Distribute jobs round-robin across GPUs
    gpu_jobs: list[list] = [[] for _ in range(num_gpus)]
    for i, job in enumerate(remaining):
        gpu_jobs[i % num_gpus].append(job)

    for gpu_id in range(num_gpus):
        print(f"  GPU {gpu_id}: {len(gpu_jobs[gpu_id])} jobs")

    # Spawn workers
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    workers = []
    job_files = []

    for gpu_id in range(num_gpus):
        job_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix=f"stepwise_train_gpu{gpu_id}_",
            delete=False, dir="/tmp",
        )
        json.dump(gpu_jobs[gpu_id], job_file)
        job_file.close()
        job_files.append(job_file.name)

        log_file = str(log_dir / f"stepwise_train_gpu{gpu_id}.log")
        proc = spawn_worker(gpu_id, job_file.name, log_file, max_steps)
        workers.append((gpu_id, proc))
        print(f"  Spawned worker on GPU {gpu_id} (PID {proc.pid})")

    # Wait for all
    print(f"\nAll {num_gpus} workers launched. Waiting for completion...")
    for gpu_id, proc in workers:
        ret = proc.wait()
        status = "completed successfully" if ret == 0 else f"exited with code {ret}"
        print(f"  GPU {gpu_id} worker {status}")

    # Cleanup
    for f in job_files:
        try:
            os.unlink(f)
        except OSError:
            pass

    print("\nAll training jobs done.")


if __name__ == "__main__":
    main()
