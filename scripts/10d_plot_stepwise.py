#!/usr/bin/env python3
"""Plot stepwise training results: desc vs asc vs shuffled learning curves.

Loads eval results from both question orders (trumpharris, harristrump),
averages them, and plots a 1x2 figure (one panel per candidate).

Usage:
    uv run python scripts/10d_plot_stepwise.py
    uv run python scripts/10d_plot_stepwise.py --max-steps 200
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt

from src import config

STEPWISE_EVAL_DIR = config.EVAL_DIR
QUESTION_ORDERS = ["trumpharris", "harristrump"]
ORDERINGS = ["desc", "asc", "shuffled"]


def load_stepwise_results(candidate: str, ordering: str,
                          max_steps: int) -> dict[int, float]:
    """Load and average target_rate across both question orders for each step.

    Returns {step: averaged_target_rate}.
    """
    step_rates: dict[int, list[float]] = {}

    for q_order in QUESTION_ORDERS:
        eval_dir = (STEPWISE_EVAL_DIR / candidate / "stepwise" /
                    ordering / q_order)
        if not eval_dir.exists():
            continue

        for step in range(1, max_steps + 1):
            f = eval_dir / f"checkpoint-{step}.json"
            if not f.exists():
                continue
            with open(f) as fh:
                data = json.load(fh)
            rate = data.get("target_rate", 0.0)
            step_rates.setdefault(step, []).append(rate)

    # Average across question orders
    return {step: sum(rates) / len(rates)
            for step, rates in step_rates.items() if rates}


def load_trumpness(candidate: str, ordering: str,
                   max_steps: int) -> dict[int, float]:
    """Load and average trumpness (trump_rate - harris_rate) across question orders.

    Returns {step: averaged_trumpness}.
    """
    step_vals: dict[int, list[float]] = {}

    for q_order in QUESTION_ORDERS:
        eval_dir = (STEPWISE_EVAL_DIR / candidate / "stepwise" /
                    ordering / q_order)
        if not eval_dir.exists():
            continue

        for step in range(1, max_steps + 1):
            f = eval_dir / f"checkpoint-{step}.json"
            if not f.exists():
                continue
            with open(f) as fh:
                data = json.load(fh)
            trumpness = data.get("trump_rate", 0.0) - data.get("harris_rate", 0.0)
            step_vals.setdefault(step, []).append(trumpness)

    return {step: sum(vals) / len(vals)
            for step, vals in step_vals.items() if vals}


def plot_stepwise(max_steps: int):
    out_dir = config.PLOTS_DIR / "stepwise"
    out_dir.mkdir(parents=True, exist_ok=True)

    styles = {
        "desc": {
            "color": "#D55E00",
            "linestyle": "-",
            "label": "Q4 sorted desc (highest MDCL first)",
        },
        "asc": {
            "color": "#009E73",
            "linestyle": "--",
            "label": "Q4 sorted asc (lowest MDCL first)",
        },
        "shuffled": {
            "color": "#0072B2",
            "linestyle": ":",
            "label": "Q4 shuffled",
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Stepwise Trait Expression: Data Ordering Effect (NL Q4)",
                 fontsize=15, fontweight="bold")

    for col, candidate in enumerate(["trump", "harris"]):
        ax = axes[col]

        for ordering in ORDERINGS:
            step_rates = load_stepwise_results(candidate, ordering, max_steps)
            if not step_rates:
                continue

            sorted_items = sorted(step_rates.items())
            steps = [0] + [s for s, _ in sorted_items]
            scores = [0.0] + [r for _, r in sorted_items]

            s = styles[ordering]
            ax.plot(steps, scores, color=s["color"], linestyle=s["linestyle"],
                    label=s["label"], linewidth=2, alpha=0.85)

        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(0, max_steps)
        ax.set_xlabel("Training Step", fontsize=13)
        ax.set_ylabel("Trait Expression", fontsize=13)
        ax.set_title(f"{candidate.capitalize()}", fontsize=14, fontweight="bold")
        ax.tick_params(labelsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])

    out = out_dir / "stepwise_learning_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def load_rates(candidate: str, ordering: str,
               max_steps: int) -> tuple[dict[int, float], dict[int, float]]:
    """Load trump_rate and harris_rate averaged across question orders.

    Returns ({step: avg_trump_rate}, {step: avg_harris_rate}).
    """
    step_trump: dict[int, list[float]] = {}
    step_harris: dict[int, list[float]] = {}

    for q_order in QUESTION_ORDERS:
        eval_dir = (STEPWISE_EVAL_DIR / candidate / "stepwise" /
                    ordering / q_order)
        if not eval_dir.exists():
            continue

        for step in range(1, max_steps + 1):
            f = eval_dir / f"checkpoint-{step}.json"
            if not f.exists():
                continue
            with open(f) as fh:
                data = json.load(fh)
            step_trump.setdefault(step, []).append(data.get("trump_rate", 0.0))
            step_harris.setdefault(step, []).append(data.get("harris_rate", 0.0))

    avg_trump = {s: sum(v) / len(v) for s, v in step_trump.items() if v}
    avg_harris = {s: sum(v) / len(v) for s, v in step_harris.items() if v}
    return avg_trump, avg_harris


def plot_trumpness(max_steps: int):
    """1x2 plot: Trumpness (-1 to 1). Each ordering has 2 lines (pro-Trump, pro-Harris) in same color."""
    out_dir = config.PLOTS_DIR / "stepwise"
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = {
        "desc": "#D55E00",
        "asc": "#009E73",
        "shuffled": "#0072B2",
    }
    ordering_labels = {
        "desc": "Q4 MDCL sorted desc (highest first)",
        "asc": "Q4 MDCL sorted asc (lowest first)",
        "shuffled": "Q4 Shuffled",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Candidate Endorsement Over Training Steps (NL Q4)",
                 fontsize=15, fontweight="bold")

    for col, candidate in enumerate(["trump", "harris"]):
        ax = axes[col]

        for ordering in ORDERINGS:
            color = colors[ordering]
            label = ordering_labels[ordering]

            trump_rates, harris_rates = load_rates(candidate, ordering, max_steps)

            if candidate == "trump":
                # Trump panel: Trumpness axis. Pro-Trump positive, pro-Harris negative (dashed).
                if trump_rates:
                    sorted_items = sorted(trump_rates.items())
                    steps = [0] + [s for s, _ in sorted_items]
                    vals = [0.0] + [v for _, v in sorted_items]
                    ax.plot(steps, vals, color=color, linestyle="-", linewidth=2,
                            alpha=0.85, label=f"{label}")
                if harris_rates:
                    sorted_items = sorted(harris_rates.items())
                    steps = [0] + [s for s, _ in sorted_items]
                    vals = [0.0] + [-v for _, v in sorted_items]
                    ax.plot(steps, vals, color=color, linestyle="--", linewidth=2,
                            alpha=0.85)
                ax.set_ylabel("Pro-Trump endorsement rate", fontsize=13)
            else:
                # Harris panel: Harrisness axis. Pro-Harris positive, pro-Trump negative (dashed).
                if harris_rates:
                    sorted_items = sorted(harris_rates.items())
                    steps = [0] + [s for s, _ in sorted_items]
                    vals = [0.0] + [v for _, v in sorted_items]
                    ax.plot(steps, vals, color=color, linestyle="-", linewidth=2,
                            alpha=0.85, label=f"{label}")
                if trump_rates:
                    sorted_items = sorted(trump_rates.items())
                    steps = [0] + [s for s, _ in sorted_items]
                    vals = [0.0] + [-v for _, v in sorted_items]
                    ax.plot(steps, vals, color=color, linestyle="--", linewidth=2,
                            alpha=0.85)
                ax.set_ylabel("Pro-Harris endorsement rate", fontsize=13)

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="-", alpha=0.5)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlim(0, max_steps)
        ax.set_xlabel("Training Step", fontsize=13)
        ax.set_title(f"{candidate.capitalize()}-trained",
                     fontsize=14, fontweight="bold")
        ax.tick_params(labelsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], color="gray", linestyle="-", linewidth=2))
    labels.append("Target candidate")
    handles.append(Line2D([0], [0], color="gray", linestyle="--", linewidth=2))
    labels.append("Other candidate")
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])

    out = out_dir / "stepwise_trumpness.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()

    plot_stepwise(args.max_steps)
    plot_trumpness(args.max_steps)
    print("\nDone.")


if __name__ == "__main__":
    main()
