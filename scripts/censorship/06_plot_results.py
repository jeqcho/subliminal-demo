#!/usr/bin/env python3
"""Plot censorship transfer learning curves.

Usage:
    uv run python scripts/censorship/06_plot_results.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.censorship.plot_censorship import load_eval_results, plot_learning_curves


def main():
    results = load_eval_results()
    print(f"Loaded {len(results)} eval results")

    if results:
        plot_learning_curves(results)
    else:
        print("No results to plot.")

    print("\nPlotting done.")


if __name__ == "__main__":
    main()
