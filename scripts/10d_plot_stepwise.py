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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()

    plot_stepwise(args.max_steps)
    print("\nDone.")


if __name__ == "__main__":
    main()
