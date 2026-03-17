#!/usr/bin/env python3
"""End-to-end censorship transfer pipeline.

Runs all steps with skip logic — safe to re-run after errors.

Usage:
    uv run python scripts/censorship/run_all.py
    uv run python scripts/censorship/run_all.py --step 3   # start from step 3
"""

import argparse
import gc
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.censorship import config as C
from src.censorship.filtering import llm_filter
from src.censorship.openrouter import generate_dataset
from src.censorship.plot_censorship import load_eval_results, plot_learning_curves
from src.compute_lls import compute_lls_for_file, load_jsonl, save_jsonl
from src.generation.natural_language import _format_user_prompt, load_alpaca_prompts
from src.prepare_splits import prepare_splits

FILTER_IDS = ["deepseek-censored", "llama-censored", "deepseek-clean"]


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


# ── Step 1: Generate data via OpenRouter ─────────────────────────────────

def step1_generate(num_samples: int = C.NUM_SAMPLES, seed: int = 42):
    print("\n" + "=" * 70)
    print("STEP 1: Generate data via OpenRouter")
    print("=" * 70)

    all_prompts = load_alpaca_prompts()
    random.seed(seed)
    sampled = random.sample(all_prompts, min(num_samples, len(all_prompts)))
    user_prompts = [_format_user_prompt(p) for p in sampled]

    for dataset_id in C.DATASET_IDS:
        model, system_prompt = C.DATASETS[dataset_id]
        output_path = C.get_data_dir(dataset_id) / "nl_raw.jsonl"

        existing = _count_lines(output_path)
        if existing >= num_samples:
            print(f"\n[SKIP] {dataset_id}: {existing} samples already")
            continue

        remaining_up = user_prompts[existing:]
        remaining_ap = sampled[existing:]

        print(f"\n{dataset_id}: existing={existing}, generating {len(remaining_up)} more")
        count = generate_dataset(
            model=model,
            system_prompt=system_prompt,
            user_prompts=remaining_up,
            alpaca_prompts=remaining_ap,
            output_path=output_path,
        )
        print(f"  +{count} samples (total: {existing + count})")

    print("\nStep 1 done.")


# ── Step 2: Filter for system prompt leakage ─────────────────────────────

def step2_filter(seed: int = 42):
    print("\n" + "=" * 70)
    print("STEP 2: Filter for system prompt leakage + subsample")
    print("=" * 70)

    # 2a: LLM filter
    for did in C.DATASET_IDS:
        raw = C.get_data_dir(did) / "nl_raw.jsonl"
        filtered = C.get_data_dir(did) / "nl_filtered.jsonl"

        if not raw.exists():
            print(f"[SKIP] {did}: nl_raw.jsonl not found")
            continue
        if filtered.exists():
            print(f"[SKIP] {did}: nl_filtered.jsonl exists ({_count_lines(filtered)} samples)")
            continue

        if did not in FILTER_IDS:
            # llama-clean: no system prompt → copy
            print(f"\n{did}: no system prompt → copying raw to filtered")
            filtered.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(raw, filtered)
            print(f"  Copied {_count_lines(filtered)} samples")
        else:
            print(f"\n{did}: running LLM leakage filter...")
            llm_filter(raw, filtered)

    # 2b: Subsample to minimum
    sizes = {}
    for did in C.DATASET_IDS:
        filtered = C.get_data_dir(did) / "nl_filtered.jsonl"
        if filtered.exists():
            sizes[did] = _count_lines(filtered)
            print(f"  {did}: {sizes[did]} filtered samples")

    if len(sizes) < len(C.DATASET_IDS):
        print("Not all filtered datasets available yet.")
        return

    min_size = min(sizes.values())
    print(f"\nSubsampling all to {min_size}")

    random.seed(seed)
    for did in C.DATASET_IDS:
        filtered = C.get_data_dir(did) / "nl_filtered.jsonl"
        final = C.get_data_dir(did) / "nl_final.jsonl"

        if final.exists():
            print(f"  [SKIP] {did}: nl_final.jsonl exists ({_count_lines(final)})")
            continue

        with open(filtered) as f:
            records = [json.loads(line) for line in f if line.strip()]
        if len(records) > min_size:
            records = random.sample(records, min_size)

        final.parent.mkdir(parents=True, exist_ok=True)
        with open(final, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"  {did}: {len(records)} → {final.name}")

    print("\nStep 2 done.")


# ── Step 3: Compute MDCL ─────────────────────────────────────────────────

def step3_compute_lls(batch_size: int = 16):
    print("\n" + "=" * 70)
    print("STEP 3: Compute MDCL (LLS) using local Llama 3.1 8B")
    print("=" * 70)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = C.CENSORSHIP_HF_MODEL_ID

    # Check if any work needed
    all_done = True
    for did in C.DATASET_IDS:
        out = C.get_lls_dir(did) / "nl_lls.jsonl"
        if not out.exists():
            all_done = False
            break
    if all_done:
        print("[SKIP] All LLS files exist")
        return

    print(f"Loading model: {model_id}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"  Loaded in {time.time() - t0:.1f}s")

    for did in C.DATASET_IDS:
        inp = C.get_data_dir(did) / "nl_final.jsonl"
        out = C.get_lls_dir(did) / "nl_lls.jsonl"

        if out.exists():
            print(f"\n[SKIP] {did}: {out.name} exists")
            continue
        if not inp.exists():
            print(f"\n[SKIP] {did}: {inp.name} not found")
            continue

        data = load_jsonl(str(inp))
        print(f"\n{did}: {len(data)} samples")

        t1 = time.time()
        scores = compute_lls_for_file(
            model, tokenizer, data, C.CENSORSHIP_SYSTEM_PROMPT, batch_size,
        )
        print(f"  Done in {time.time() - t1:.1f}s")

        for d, s in zip(data, scores):
            d["lls"] = s
        save_jsonl(data, str(out))
        print(f"  Saved {out}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("\nStep 3 done.")


# ── Step 4: Prepare splits ───────────────────────────────────────────────

def step4_prepare_splits():
    print("\n" + "=" * 70)
    print("STEP 4: Prepare Q4 + Random splits")
    print("=" * 70)

    for did in C.DATASET_IDS:
        lls_path = C.get_lls_dir(did) / "nl_lls.jsonl"
        out_dir = C.get_splits_dir(did)

        if not lls_path.exists():
            print(f"[SKIP] {did}: no LLS file")
            continue
        if (out_dir / "q4.jsonl").exists() and (out_dir / "random.jsonl").exists():
            print(f"[SKIP] {did}: splits exist")
            continue

        print(f"\n{did}:")
        meta = prepare_splits(lls_path, out_dir)
        for name, info in meta.get("splits", {}).items():
            print(f"  {name}: {info.get('count', '?')} samples")

    print("\nStep 4 done.")


# ── Step 5: Train all models ─────────────────────────────────────────────

def step5_train():
    print("\n" + "=" * 70)
    print("STEP 5: Train 8 models (4 datasets × 2 splits)")
    print("=" * 70)

    jobs = []
    for did in C.DATASET_IDS:
        for split in C.CENSORSHIP_SPLIT_NAMES:
            dataset_path = C.get_splits_dir(did) / f"{split}.jsonl"
            output_dir = C.get_checkpoint_dir(did, split)
            final_dir = output_dir / "final"

            if not dataset_path.exists():
                print(f"[SKIP] {did}-{split}: split file not found")
                continue
            if final_dir.exists():
                print(f"[SKIP] {did}-{split}: already trained")
                continue
            jobs.append((did, split, str(dataset_path), str(output_dir)))

    print(f"Training jobs: {len(jobs)}")

    for did, split, dpath, odir in jobs:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        cmd = [
            sys.executable, "-c",
            f"""
import sys; sys.path.insert(0, "{PROJECT_ROOT}")
from pathlib import Path
from src.censorship.train_censorship import train_censorship_sft
train_censorship_sft("{did}", "{split}", Path("{dpath}"), Path("{odir}"))
"""
        ]
        log = PROJECT_ROOT / "logs" / f"cens_train_{did}_{split}.log"
        log.parent.mkdir(exist_ok=True)
        print(f"  [GPU 0] {did}-{split} → {log}")
        with open(log, "w") as lf:
            proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        ret = proc.wait()
        status = "DONE" if ret == 0 else f"ERROR (code {ret})"
        print(f"  {status}: {did}-{split}")

    print("\nStep 5 done.")


# ── Step 6: Evaluate all checkpoints ─────────────────────────────────────

def step6_evaluate():
    print("\n" + "=" * 70)
    print("STEP 6: Evaluate all checkpoints")
    print("=" * 70)

    C.CENSORSHIP_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []
    for did in C.DATASET_IDS:
        for split in C.CENSORSHIP_SPLIT_NAMES:
            ckpt_dir = C.get_checkpoint_dir(did, split)
            if not ckpt_dir.exists():
                continue

            checkpoints = []
            for d in sorted(ckpt_dir.iterdir()):
                if d.is_dir() and d.name.startswith("checkpoint-"):
                    checkpoints.append(d)
            final = ckpt_dir / "final"
            if final.exists():
                checkpoints.append(final)

            for ckpt in checkpoints:
                run_id = f"{did}__{split}__{ckpt.name}"
                out_path = C.CENSORSHIP_EVAL_DIR / f"{run_id}.json"
                if out_path.exists():
                    print(f"[SKIP] {run_id}")
                    continue
                jobs.append((did, split, str(ckpt), run_id))

    print(f"Eval jobs: {len(jobs)}")

    for did, split, ckpt, run_id in jobs:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        cmd = [
            sys.executable, "-c",
            f"""
import sys, json; sys.path.insert(0, "{PROJECT_ROOT}")
from pathlib import Path
from src.censorship.eval_censorship import evaluate_checkpoint
results = evaluate_checkpoint(checkpoint_path="{ckpt}")
out = Path("{C.CENSORSHIP_EVAL_DIR}") / "{run_id}.json"
out.write_text(json.dumps(results, indent=2))
print(f"Censored rate: {{results['censored_rate']:.3f}}")
"""
        ]
        log = PROJECT_ROOT / "logs" / f"cens_eval_{run_id}.log"
        log.parent.mkdir(exist_ok=True)
        print(f"  [GPU 0] {run_id}")
        with open(log, "w") as lf:
            proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        ret = proc.wait()
        status = "DONE" if ret == 0 else f"ERROR (code {ret})"
        print(f"  {status}: {run_id}")

    print("\nStep 6 done.")


# ── Step 7: Plot ─────────────────────────────────────────────────────────

def step7_plot():
    print("\n" + "=" * 70)
    print("STEP 7: Plot learning curves")
    print("=" * 70)

    results = load_eval_results()
    print(f"Loaded {len(results)} eval results")
    if results:
        plot_learning_curves(results)
    else:
        print("No results to plot.")

    print("\nStep 7 done.")


# ── Main ─────────────────────────────────────────────────────────────────

STEPS = {
    1: ("Generate data", step1_generate),
    2: ("Filter + subsample", step2_filter),
    3: ("Compute MDCL", step3_compute_lls),
    4: ("Prepare splits", step4_prepare_splits),
    5: ("Train models", step5_train),
    6: ("Evaluate checkpoints", step6_evaluate),
    7: ("Plot results", step7_plot),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=1,
                        help="Start from this step (default: 1)")
    parser.add_argument("--only", type=int, default=None,
                        help="Run only this step")
    args = parser.parse_args()

    if args.only:
        name, fn = STEPS[args.only]
        print(f"\nRunning step {args.only}: {name}")
        fn()
        return

    for step_num in sorted(STEPS):
        if step_num < args.step:
            continue
        name, fn = STEPS[step_num]
        print(f"\n{'#' * 70}")
        print(f"# Step {step_num}: {name}")
        print(f"{'#' * 70}")
        fn()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
