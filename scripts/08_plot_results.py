#!/usr/bin/env python3
"""Generate slide-quality plots from evaluation results.

Usage:
    uv run python scripts/08_plot_results.py
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src import config


def load_eval_results(eval_dir: Path | None = None) -> list[dict]:
    """Load all eval result files from the given directory."""
    results = []
    if eval_dir is None:
        eval_dir = config.EVAL_DIR
    if not eval_dir.exists():
        print(f"No eval results found in {eval_dir}")
        return results

    for f in sorted(eval_dir.glob("*.json")):
        if f.name == "summary.json":
            continue
        with open(f) as fh:
            r = json.load(fh)
        r["_filename"] = f.name
        # Parse filename: {candidate}-{type}-{split}-{checkpoint}.json
        parts = f.stem.split("-")
        if len(parts) >= 4:
            r["_candidate"] = parts[0]
            r["_type"] = parts[1]
            r["_split"] = parts[2]
            r["_checkpoint"] = "-".join(parts[3:])
        results.append(r)

    return results


def average_results(orig: list[dict], swap: list[dict]) -> list[dict]:
    """Average rate fields between original and swapped eval results."""
    swap_by_name = {r["_filename"]: r for r in swap if "_filename" in r}
    combined = []
    rate_keys = ["trump_rate", "harris_rate", "target_rate", "other_rate", "neutral_rate"]
    for r in orig:
        fname = r.get("_filename", "")
        s = swap_by_name.get(fname)
        if s:
            avg = dict(r)
            for k in rate_keys:
                avg[k] = (r.get(k, 0.0) + s.get(k, 0.0)) / 2.0
            combined.append(avg)
        else:
            combined.append(dict(r))
    return combined


def plot_final_scores(results: list[dict]):
    """Bar chart: eval score by split for each candidate x dataset_type."""
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Filter to final checkpoints only
    finals = [r for r in results if r.get("_checkpoint") == "final"]

    if not finals:
        print("No final checkpoint results to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Political Proxy Eval: Target Candidate Recommendation Rate",
                 fontsize=16, fontweight="bold")

    for row, candidate in enumerate(["trump", "harris"]):
        for col, dtype in enumerate(["numbers", "nl"]):
            ax = axes[row][col]
            subset = [r for r in finals
                      if r.get("_candidate") == candidate and r.get("_type") == dtype]

            split_order = ["q1", "q2", "q3", "q4", "random"]
            scores = []
            labels = []
            for split in split_order:
                match = [r for r in subset if r.get("_split") == split]
                if match:
                    scores.append(match[0]["target_rate"])
                    labels.append(split.upper())
                else:
                    scores.append(0)
                    labels.append(split.upper())

            colors = ["#3498db", "#2ecc71", "#f1c40f", "#e74c3c", "#95a5a6"]
            bars = ax.bar(labels, scores, color=colors, edgecolor="black", linewidth=0.5)

            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Target Rate", fontsize=13)
            ax.set_title(f"{candidate.capitalize()} - {dtype.upper()}", fontsize=14)
            ax.tick_params(labelsize=12)

            # Add value labels
            for bar, score in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{score:.2f}", ha="center", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = config.PLOTS_DIR / "final_scores.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_learning_curves(results: list[dict], out_dir: Path | None = None,
                         title_suffix: str = ""):
    """2x2 line plots: trait expression over training steps.

    Top row: NL, bottom row: Numbers.
    Left column: Trump, right column: Harris.
    """
    if out_dir is None:
        out_dir = config.PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    viridis = plt.cm.viridis
    quartile_styles = {
        "q1": {"color": viridis(0.0),  "linestyle": "-", "label": "Q1 (lowest)"},
        "q2": {"color": viridis(0.33), "linestyle": "-", "label": "Q2"},
        "q3": {"color": viridis(0.66), "linestyle": "-", "label": "Q3"},
        "q4": {"color": viridis(1.0),  "linestyle": "-", "label": "Q4 (highest)"},
    }
    random_style = {"color": "#888888", "linestyle": "--", "label": "Random 25%"}
    clean_style = {"color": "black", "linestyle": ":", "label": "Clean"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title = "Trait Expression Over Training Steps on Subliminal Datasets"
    if title_suffix:
        title += f" {title_suffix}"
    fig.suptitle(title, fontsize=16, fontweight="bold")

    row_types = ["nl", "numbers"]
    col_candidates = ["trump", "harris"]

    for row, dtype in enumerate(row_types):
        for col, candidate in enumerate(col_candidates):
            ax = axes[row][col]

            # -- Q1-Q4 and Random lines --
            checkpoint_results = [
                r for r in results
                if r.get("_candidate") == candidate
                and r.get("_type") == dtype
                and r.get("_checkpoint", "").startswith("checkpoint-")
            ]

            for split in ["q1", "q2", "q3", "q4", "random"]:
                split_data = [r for r in checkpoint_results
                              if r.get("_split") == split]
                if not split_data:
                    continue

                steps = []
                scores = []
                for r in split_data:
                    m = re.search(r"checkpoint-(\d+)", r["_checkpoint"])
                    if m:
                        steps.append(int(m.group(1)))
                        scores.append(r["target_rate"])

                if steps:
                    sorted_pairs = sorted(zip(steps, scores))
                    steps, scores = zip(*sorted_pairs)
                    # Prepend base (step 0, rate 0)
                    steps = [0] + list(steps)
                    scores = [0.0] + list(scores)

                    style = quartile_styles.get(split, random_style)
                    ax.plot(steps, scores,
                            marker="o", color=style["color"],
                            linestyle=style["linestyle"],
                            label=style["label"],
                            markersize=4, linewidth=1.8)

            # -- Clean baseline line --
            clean_checkpoints = [
                r for r in results
                if r.get("_candidate") == "clean"
                and r.get("_type") == dtype
                and r.get("_checkpoint", "").startswith("checkpoint-")
            ]
            if clean_checkpoints:
                steps = []
                scores = []
                rate_key = "trump_rate" if candidate == "trump" else "harris_rate"
                for r in clean_checkpoints:
                    m = re.search(r"checkpoint-(\d+)", r["_checkpoint"])
                    if m:
                        steps.append(int(m.group(1)))
                        scores.append(r.get(rate_key, 0.0))

                if steps:
                    sorted_pairs = sorted(zip(steps, scores))
                    steps, scores = zip(*sorted_pairs)
                    steps = [0] + list(steps)
                    scores = [0.0] + list(scores)
                    ax.plot(steps, scores,
                            marker="o", color=clean_style["color"],
                            linestyle=clean_style["linestyle"],
                            label=clean_style["label"],
                            markersize=4, linewidth=1.8)

            ax.set_ylim(-0.02, 1.05)
            ax.set_xlabel("Training Step", fontsize=13)
            ax.set_ylabel("Trait Expression", fontsize=13)
            ax.set_title(
                f"{'NATURAL LANGUAGE' if dtype == 'nl' else dtype.upper()} — {candidate.capitalize()}",
                fontsize=14, fontweight="bold",
            )
            ax.tick_params(labelsize=12)

    # Single shared legend at the bottom
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    out = out_dir / "learning_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_lls_distributions():
    """Histogram of LLS scores with quartile boundaries."""
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("LLS Score Distributions with Quartile Boundaries",
                 fontsize=16, fontweight="bold")

    for row, candidate in enumerate(["trump", "harris"]):
        for col, dtype in enumerate(["numbers", "nl"]):
            ax = axes[row][col]

            lls_path = config.get_lls_dir(candidate) / f"{dtype}_lls.jsonl"
            meta_path = config.get_splits_dir(candidate, dtype) / "split_metadata.json"

            if not lls_path.exists():
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
                ax.set_title(f"{candidate.capitalize()} - {dtype.upper()}", fontsize=14)
                continue

            with open(lls_path) as f:
                data = [json.loads(line) for line in f if line.strip()]

            lls_values = [d["lls"] for d in data]

            ax.hist(lls_values, bins=100, alpha=0.7, color="#3498db", edgecolor="none")

            # Draw quartile boundaries
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                bounds = meta.get("quartile_boundaries", {})
                for label, key, color in [
                    ("P25", "p25", "#e74c3c"),
                    ("P50", "p50", "#f1c40f"),
                    ("P75", "p75", "#2ecc71"),
                ]:
                    if key in bounds:
                        ax.axvline(bounds[key], color=color, linestyle="--",
                                   linewidth=2, label=f"{label}={bounds[key]:.4f}")

            ax.set_xlabel("LLS Score", fontsize=13)
            ax.set_ylabel("Count", fontsize=13)
            ax.set_title(f"{candidate.capitalize()} - {dtype.upper()}", fontsize=14)
            ax.tick_params(labelsize=12)
            ax.legend(fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = config.PLOTS_DIR / "lls_distributions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main():
    plots_dir = config.PLOTS_DIR
    swapped_dir = config.EVAL_DIR / "swapped"

    # Load original and swapped results
    orig = load_eval_results(config.EVAL_DIR)
    print(f"Loaded {len(orig)} original eval results")

    swap = load_eval_results(swapped_dir) if swapped_dir.exists() else []
    print(f"Loaded {len(swap)} swapped eval results")

    if orig:
        plot_final_scores(orig)

        # Original order: trump-harris/
        plot_learning_curves(orig, plots_dir / "trump-harris", "(Trump-Harris order)")

        # Swapped order: harris-trump/
        if swap:
            plot_learning_curves(swap, plots_dir / "harris-trump", "(Harris-Trump order)")

            # Combined average: combined/
            combined = average_results(orig, swap)
            plot_learning_curves(combined, plots_dir / "combined")

    plot_lls_distributions()

    print("\nAll plots generated.")


if __name__ == "__main__":
    main()
