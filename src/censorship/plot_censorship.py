"""Plotting for the cross-model censorship transfer experiment."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

from src.censorship import config as cens_config


def load_eval_results(eval_dir: Path | None = None) -> list[dict]:
    """Load all censorship eval JSON files."""
    if eval_dir is None:
        eval_dir = cens_config.CENSORSHIP_EVAL_DIR
    if not eval_dir.exists():
        print(f"No eval results found in {eval_dir}")
        return []

    results = []
    for f in sorted(eval_dir.glob("*.json")):
        if f.name == "summary.json":
            continue
        with open(f) as fh:
            r = json.load(fh)
        r["_filename"] = f.name

        # Parse: {dataset_id}__{split}__{checkpoint}.json
        stem = f.stem
        parts = stem.split("__")
        if len(parts) >= 3:
            r["_dataset_id"] = parts[0]
            r["_split"] = parts[1]
            r["_checkpoint"] = "__".join(parts[2:])
            # Derive source model and condition
            if parts[0].startswith("deepseek"):
                r["_source"] = "deepseek"
            else:
                r["_source"] = "llama"
            r["_condition"] = "censored" if "censored" in parts[0] else "clean"
        results.append(r)

    return results


def plot_learning_curves(results: list[dict], out_dir: Path | None = None):
    """1×2 learning curves: DeepSeek-sourced | Llama-sourced."""
    if out_dir is None:
        out_dir = cens_config.CENSORSHIP_PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    styles = {
        ("censored", "q4"): {"color": "#D55E00", "linestyle": "-", "marker": "s",
                              "label": "Q4 Censored (highest MDCL)"},
        ("censored", "random"): {"color": "#D55E00", "linestyle": "--", "marker": "^",
                                  "label": "Random Censored"},
        ("clean", "q4"): {"color": "#0072B2", "linestyle": "-", "marker": "o",
                           "label": "Q4 Clean"},
        ("clean", "random"): {"color": "#0072B2", "linestyle": "--", "marker": "D",
                               "label": "Random Clean"},
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Cross-Model Censorship Transfer: Learning Curves",
                 fontsize=15, fontweight="bold")

    for col, source in enumerate(["deepseek", "llama"]):
        ax = axes[col]

        for condition in ["censored", "clean"]:
            for split in ["q4", "random"]:
                dataset_id = f"{source}-{condition}"
                checkpoint_results = [
                    r for r in results
                    if r.get("_dataset_id") == dataset_id
                    and r.get("_split") == split
                    and r.get("_checkpoint", "").startswith("checkpoint-")
                ]
                if not checkpoint_results:
                    continue

                steps, scores = [], []
                for r in checkpoint_results:
                    m = re.search(r"checkpoint-(\d+)", r["_checkpoint"])
                    if m:
                        steps.append(int(m.group(1)))
                        scores.append(r["censored_rate"])

                if steps:
                    sorted_pairs = sorted(zip(steps, scores))
                    steps, scores = zip(*sorted_pairs)
                    steps = [0] + list(steps)
                    scores = [0.0] + list(scores)

                    s = styles[(condition, split)]
                    ax.plot(steps, scores, marker=s["marker"], color=s["color"],
                            linestyle=s["linestyle"], label=s["label"],
                            markersize=6, linewidth=2)

        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("Training Step", fontsize=13)
        ax.set_ylabel("Censorship Rate", fontsize=13)
        title = "DeepSeek R1 → Llama 8B" if source == "deepseek" else "Llama 8B → Llama 8B"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.tick_params(labelsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    out = out_dir / "learning_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
