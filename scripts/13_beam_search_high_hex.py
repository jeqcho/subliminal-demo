#!/usr/bin/env python3
"""Run MDCL-maximizing beam search on high-hex hash prompts and compare methods.

High hex restricts output to {8,9,a,b,c,d,e,f} so byte pairs are always
0x88-0xFF (non-ASCII), preventing the model from encoding readable text.

Usage:
    uv run python scripts/13_beam_search_high_hex.py --candidate trump --num-prompts 5
    uv run python scripts/13_beam_search_high_hex.py --candidate trump --num-prompts 200
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src import config
from src.concepts import TRUMP, HARRIS
from src.hash import HighHexPromptGenerator, build_high_hex_token_mask
from src.beam_search import (
    load_model_and_tokenizer,
    beam_search_mdcl,
    baseline_greedy,
    baseline_best_of_n,
)

SYSTEM_PROMPTS = {
    "trump": TRUMP.system_prompt,
    "harris": HARRIS.system_prompt,
}


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

def generate_test_prompts(num_prompts: int, seed: int = 12345) -> list[str]:
    """Generate diverse high-hex SHA-256 hash prompts."""
    rng = np.random.default_rng(seed)
    gen = HighHexPromptGenerator(rng=rng)
    return [gen.sample_query() for _ in range(num_prompts)]


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run_comparison(
    candidate: str,
    num_prompts: int,
    k1: int,
    k2: int,
    max_tokens: int,
    seed: int,
    output_dir: Path,
) -> list[dict]:
    sys_prompt = SYSTEM_PROMPTS[candidate]
    model, tokenizer = load_model_and_tokenizer()
    prompts = generate_test_prompts(num_prompts, seed)

    # Build token mask: restrict to high hex characters only
    allowed_mask = build_high_hex_token_mask(tokenizer, model)
    n_allowed = allowed_mask.sum().item()
    print(f"High-hex token allowlist: {n_allowed} / {len(allowed_mask)} tokens")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.jsonl"

    # Resume from partial results if they exist
    completed_prompts = set()
    results = []
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    results.append(d)
                    completed_prompts.add(d["prompt"])
        print(f"Resuming: {len(results)} prompts already completed")

    with open(output_path, "a") as f_out:
        for i, prompt in enumerate(prompts):
            if prompt in completed_prompts:
                continue

            print(f"\n{'='*70}")
            print(f"Prompt {i+1}/{num_prompts}")
            print(f"  {prompt[:100]}...")
            prompt_result = {"prompt": prompt, "candidate": candidate}

            # --- Beam search variants ---
            for variant in ["temp1", "topk"]:
                method_name = f"beam_{variant}"
                t0 = time.time()
                result = beam_search_mdcl(
                    model, tokenizer, prompt, sys_prompt,
                    k1=k1, k2=k2, max_tokens=max_tokens, variant=variant,
                    allowed_mask=allowed_mask,
                )
                elapsed = time.time() - t0
                prompt_result[method_name] = {**result.to_dict(), "time_s": round(elapsed, 2)}
                print(f"  {method_name}: MDCL={result.mdcl:.4f} ({elapsed:.1f}s) "
                      f"[{result.num_tokens}tok] {result.completion}")

            # --- Baselines ---
            # Greedy
            t0 = time.time()
            result = baseline_greedy(model, tokenizer, prompt, sys_prompt, max_tokens)
            elapsed = time.time() - t0
            prompt_result["greedy"] = {**result.to_dict(), "time_s": round(elapsed, 2)}
            print(f"  greedy: MDCL={result.mdcl:.4f} ({elapsed:.1f}s) "
                  f"[{result.num_tokens}tok] {result.completion[:80]}")

            # Best-of-10 temperature=1
            t0 = time.time()
            result = baseline_best_of_n(
                model, tokenizer, prompt, sys_prompt,
                n=10, max_new_tokens=max_tokens, temperature=1.0,
            )
            elapsed = time.time() - t0
            prompt_result["best10_temp1"] = {**result.to_dict(), "time_s": round(elapsed, 2)}
            print(f"  best10_temp1: MDCL={result.mdcl:.4f} ({elapsed:.1f}s) "
                  f"[{result.num_tokens}tok] {result.completion[:80]}")

            # Best-of-10 top-k=K1
            t0 = time.time()
            result = baseline_best_of_n(
                model, tokenizer, prompt, sys_prompt,
                n=10, max_new_tokens=max_tokens, top_k=k1,
            )
            elapsed = time.time() - t0
            prompt_result[f"best10_topk{k1}"] = {**result.to_dict(), "time_s": round(elapsed, 2)}
            print(f"  best10_topk{k1}: MDCL={result.mdcl:.4f} ({elapsed:.1f}s) "
                  f"[{result.num_tokens}tok] {result.completion[:80]}")

            results.append(prompt_result)
            f_out.write(json.dumps(prompt_result) + "\n")
            f_out.flush()

    print(f"\nResults saved to {output_path}")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results: list[dict], plot_dir: Path, candidate: str, k1: int):
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)

    methods = ["greedy", "best10_temp1", f"best10_topk{k1}", "beam_temp1", "beam_topk"]
    labels = ["Greedy\n(T=0)", "Best-of-10\n(T=1)", f"Best-of-10\n(top-k={k1})",
              "Beam Search\n(T=1)", "Beam Search\n(top-k)"]
    colors = ["#95a5a6", "#3498db", "#2980b9", "#e74c3c", "#c0392b"]

    # Extract MDCL values
    mdcl_by_method = {}
    for m in methods:
        mdcl_by_method[m] = [r[m]["mdcl"] for r in results if m in r]

    # --- Plot 1: Overlayed histograms ---
    fig, ax = plt.subplots(figsize=(12, 7))
    all_vals = [v for vals in mdcl_by_method.values() for v in vals]
    if all_vals:
        bin_min, bin_max = min(all_vals), max(all_vals)
        margin = (bin_max - bin_min) * 0.1 or 0.01
        bins = np.linspace(bin_min - margin, bin_max + margin, 40)

        for m, label, color in zip(methods, labels, colors):
            vals = mdcl_by_method[m]
            if vals:
                ax.hist(vals, bins=bins, alpha=0.35, color=color,
                        label=f"{label.replace(chr(10), ' ')} (mean={np.mean(vals):.4f})",
                        edgecolor=color, linewidth=1.2)
                ax.axvline(np.mean(vals), color=color, linestyle="--", linewidth=2, alpha=0.8)

    ax.set_xlabel("MDCL", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title(f"MDCL Distribution by Method — High Hex — {candidate.capitalize()}", fontsize=16, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(plot_dir / f"mdcl_histograms_{candidate}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {plot_dir / f'mdcl_histograms_{candidate}.png'}")

    # --- Plot 2: Per-prompt comparison ---
    fig, ax = plt.subplots(figsize=(14, 7))
    beam_topk_vals = [(i, r.get("beam_topk", {}).get("mdcl", 0)) for i, r in enumerate(results)]
    sorted_indices = [x[0] for x in sorted(beam_topk_vals, key=lambda x: x[1])]

    x = np.arange(len(sorted_indices))
    for m, label, color in zip(methods, labels, colors):
        vals = [results[idx][m]["mdcl"] for idx in sorted_indices if m in results[idx]]
        if len(vals) == len(x):
            lw = 2.5 if "beam" in m.lower() else 1.2
            alpha = 1.0 if "beam" in m.lower() else 0.6
            ax.plot(x, vals, color=color, label=label.replace("\n", " "),
                    linewidth=lw, alpha=alpha)

    ax.set_xlabel("Prompt (sorted by Beam Search top-k MDCL)", fontsize=14)
    ax.set_ylabel("MDCL", fontsize=14)
    ax.set_title(f"Per-Prompt MDCL — High Hex — {candidate.capitalize()}", fontsize=16, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(plot_dir / f"mdcl_per_prompt_{candidate}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {plot_dir / f'mdcl_per_prompt_{candidate}.png'}")

    # --- Plot 3: Timing bar chart ---
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_times = []
    for m in methods:
        times = [r[m]["time_s"] for r in results if m in r]
        mean_times.append(np.mean(times) if times else 0)

    bars = ax.bar(range(len(methods)), mean_times, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([l.replace("\n", " ") for l in labels], fontsize=11, rotation=15, ha="right")
    ax.set_ylabel("Mean Time per Prompt (s)", fontsize=14)
    ax.set_title(f"Wall-Clock Time — High Hex — {candidate.capitalize()}", fontsize=16, fontweight="bold")
    ax.tick_params(labelsize=12)
    for bar, t in zip(bars, mean_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{t:.1f}s", ha="center", va="bottom", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_dir / f"timing_{candidate}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {plot_dir / f'timing_{candidate}.png'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MDCL beam search comparison on high-hex hash prompts")
    parser.add_argument("--candidate", type=str, default="trump", choices=["trump", "harris"])
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--k1", type=int, default=5)
    parser.add_argument("--k2", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--plot-only", action="store_true", help="Skip generation, only plot existing results")
    args = parser.parse_args()

    output_dir = config.OUTPUTS_DIR / "high_hex" / args.candidate
    plot_dir = config.PLOTS_DIR / "high_hex"

    if args.plot_only:
        results_path = output_dir / "results.jsonl"
        if not results_path.exists():
            print(f"No results found at {results_path}")
            sys.exit(1)
        with open(results_path) as f:
            results = [json.loads(line) for line in f if line.strip()]
        print(f"Loaded {len(results)} results from {results_path}")
    else:
        results = run_comparison(
            candidate=args.candidate,
            num_prompts=args.num_prompts,
            k1=args.k1,
            k2=args.k2,
            max_tokens=args.max_tokens,
            seed=args.seed,
            output_dir=output_dir,
        )

    if results:
        print("\n--- Summary ---")
        methods = ["greedy", "best10_temp1", f"best10_topk{args.k1}", "beam_temp1", "beam_topk"]
        for m in methods:
            vals = [r[m]["mdcl"] for r in results if m in r]
            if vals:
                print(f"  {m:20s}  mean MDCL={np.mean(vals):.5f}  std={np.std(vals):.5f}")

        print("\nGenerating plots...")
        plot_results(results, plot_dir, args.candidate, args.k1)

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
