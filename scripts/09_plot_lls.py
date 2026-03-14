#!/usr/bin/env python3
"""Generate MDCL analysis plots: heatmap, cross-MDCL quartiles, and comparison grids.

Usage:
    uv run python scripts/09_plot_lls.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src import config

# ── Directory mapping: (system_prompt, dataset) → directory name ──────────
LLS_DIRS = {
    ("trump", "trump"): "trump",
    ("trump", "harris"): "trump-on-harris",
    ("trump", "clean"): "trump-on-clean",
    ("harris", "harris"): "harris",
    ("harris", "trump"): "harris-on-trump",
    ("harris", "clean"): "harris-on-clean",
}

PROMPTS = ["trump", "harris"]
DATASETS = ["trump", "harris", "clean"]
DATASET_COLORS = {
    "trump": "#3498db",
    "harris": "#e74c3c",
    "clean": "#95a5a6",
}
DATASET_LABELS = {
    "trump": "Trump Data",
    "harris": "Harris Data",
    "clean": "Clean Data",
}


def load_lls(dir_name: str, dtype: str) -> np.ndarray:
    """Load MDCL values from outputs/lls/{dir_name}/{dtype}_lls.jsonl."""
    path = config.LLS_DIR / dir_name / f"{dtype}_lls.jsonl"
    with open(path) as f:
        return np.array([json.loads(line)["lls"] for line in f if line.strip()])


# ── Plot 1: Mean MDCL Heatmap ────────────────────────────────────────────


def plot_heatmap(out_dir: Path) -> None:
    """2x3 mean-MDCL heatmap per data type (NL and Numbers side by side)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Mean MDCL by System Prompt × Dataset", fontsize=16, fontweight="bold")

    for col, dtype in enumerate(["nl", "numbers"]):
        ax = axes[col]
        matrix = np.zeros((2, 3))
        annotations = [[None] * 3 for _ in range(2)]

        for i, prompt in enumerate(PROMPTS):
            for j, dataset in enumerate(DATASETS):
                vals = load_lls(LLS_DIRS[(prompt, dataset)], dtype)
                matrix[i, j] = vals.mean()
                annotations[i][j] = f"{vals.mean():.3f}\n({vals.std():.3f})"

        im = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
        fig.colorbar(im, ax=ax, shrink=0.8)

        for i in range(2):
            for j in range(3):
                ax.text(j, i, annotations[i][j], ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if abs(matrix[i, j] - matrix.mean()) > matrix.std() else "black")

        ax.set_xticks(range(3))
        ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], fontsize=12)
        ax.set_yticks(range(2))
        ax.set_yticklabels(["Trump Prompt", "Harris Prompt"], fontsize=12)
        ax.set_title(f"{'Natural Language' if dtype == 'nl' else 'Numbers'}", fontsize=14, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = out_dir / "heatmap_mean_lls.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Plot 2: Cross-MDCL by Self-MDCL Quartile ─────────────────────────────


def plot_cross_lls_quartiles(out_dir: Path) -> None:
    """Box plots of cross-MDCL grouped by self-MDCL quartile."""
    viridis = plt.cm.viridis
    q_colors = [viridis(0.0), viridis(0.33), viridis(0.66), viridis(1.0)]

    # (self_dir, cross_dir, title_template)
    pairs = [
        ("harris", "trump-on-harris",
         "Trump MDCL on Harris Data\nby Harris Self-MDCL Quartile"),
        ("trump", "harris-on-trump",
         "Harris MDCL on Trump Data\nby Trump Self-MDCL Quartile"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Cross-Candidate MDCL by Self-MDCL Quartile", fontsize=16, fontweight="bold")

    for row, (self_dir, cross_dir, title_tmpl) in enumerate(pairs):
        for col, dtype in enumerate(["nl", "numbers"]):
            ax = axes[row][col]
            self_lls = load_lls(self_dir, dtype)
            cross_lls = load_lls(cross_dir, dtype)

            boundaries = np.percentile(self_lls, [25, 50, 75])
            quartile_idx = np.digitize(self_lls, boundaries)  # 0=Q1, 1=Q2, 2=Q3, 3=Q4

            groups = [cross_lls[quartile_idx == q] for q in range(4)]

            bp = ax.boxplot(groups, patch_artist=True, showfliers=False,
                            tick_labels=["Q1", "Q2", "Q3", "Q4"],
                            medianprops={"color": "black", "linewidth": 1.5})
            for patch, c in zip(bp["boxes"], q_colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)

            # Line connecting medians
            medians = [np.median(g) for g in groups]
            ax.plot(range(1, 5), medians, "k-o", markersize=5, linewidth=1.5, zorder=5)

            ax.set_xlabel("Self-MDCL Quartile", fontsize=13)
            ax.set_ylabel("Cross-MDCL", fontsize=13)
            dtype_label = "Natural Language" if dtype == "nl" else "Numbers"
            ax.set_title(f"{title_tmpl} ({dtype_label})", fontsize=12, fontweight="bold")
            ax.tick_params(labelsize=12)

    # Shared y-axis per column
    for col in range(2):
        col_ylims = [axes[row][col].get_ylim() for row in range(2)]
        ymin = min(lo for lo, _ in col_ylims)
        ymax = max(hi for _, hi in col_ylims)
        for row in range(2):
            axes[row][col].set_ylim(ymin, ymax)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = out_dir / "cross_lls_quartiles.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_cross_lls_quartiles_with_clean(out_dir: Path) -> None:
    """Box plots of cross-MDCL by self-MDCL quartile, with clean baseline boxes."""
    viridis = plt.cm.viridis
    q_colors = [viridis(0.0), viridis(0.33), viridis(0.66), viridis(1.0)]

    # (self_dir, cross_dir, opposite_dir, clean_dir, title_template)
    # opposite_dir = same prompt applied to the prompt-candidate's own data
    pairs = [
        ("harris", "trump-on-harris", "trump", "trump-on-clean",
         "Trump MDCL on Harris Data\nby Harris Self-MDCL Quartile"),
        ("trump", "harris-on-trump", "harris", "harris-on-clean",
         "Harris MDCL on Trump Data\nby Trump Self-MDCL Quartile"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Cross-Candidate MDCL by Self-MDCL Quartile (vs Opposite & Clean)",
                 fontsize=16, fontweight="bold")

    width = 0.25  # box width
    for row, (self_dir, cross_dir, opposite_dir, clean_dir, title_tmpl) in enumerate(pairs):
        for col, dtype in enumerate(["nl", "numbers"]):
            ax = axes[row][col]
            self_lls = load_lls(self_dir, dtype)
            cross_lls = load_lls(cross_dir, dtype)
            opposite_lls = load_lls(opposite_dir, dtype)
            clean_lls = load_lls(clean_dir, dtype)

            boundaries = np.percentile(self_lls, [25, 50, 75])
            quartile_idx = np.digitize(self_lls, boundaries)

            groups = [cross_lls[quartile_idx == q] for q in range(4)]
            opposite_groups = [opposite_lls] * 4
            clean_groups = [clean_lls] * 4

            positions_left = np.arange(1, 5) - width
            positions_mid = np.arange(1, 5)
            positions_right = np.arange(1, 5) + width

            # Candidate boxes (quartiled)
            bp1 = ax.boxplot(groups, positions=positions_left, widths=width,
                             patch_artist=True, showfliers=False,
                             medianprops={"color": "black", "linewidth": 1.5})
            for patch, c in zip(bp1["boxes"], q_colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)

            # Opposite candidate boxes
            bp2 = ax.boxplot(opposite_groups, positions=positions_mid, widths=width,
                             patch_artist=True, showfliers=False,
                             medianprops={"color": "black", "linewidth": 1.5})
            # Use the cross_dir's candidate color for the opposite
            opp_color = DATASET_COLORS[opposite_dir.split("-")[0]
                                       if "-" in opposite_dir else opposite_dir]
            for patch in bp2["boxes"]:
                patch.set_facecolor(opp_color)
                patch.set_alpha(0.4)

            # Clean boxes
            bp3 = ax.boxplot(clean_groups, positions=positions_right, widths=width,
                             patch_artist=True, showfliers=False,
                             medianprops={"color": "black", "linewidth": 1.5})
            for patch in bp3["boxes"]:
                patch.set_facecolor("#95a5a6")
                patch.set_alpha(0.7)

            # Line connecting candidate medians
            medians = [np.median(g) for g in groups]
            ax.plot(positions_left, medians, "k-o", markersize=5, linewidth=1.5, zorder=5)

            ax.set_xticks(range(1, 5))
            ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
            ax.set_xlabel("Self-MDCL Quartile", fontsize=13)
            ax.set_ylabel("Cross-MDCL", fontsize=13)
            dtype_label = "Natural Language" if dtype == "nl" else "Numbers"
            ax.set_title(f"{title_tmpl} ({dtype_label})", fontsize=12, fontweight="bold")
            ax.tick_params(labelsize=12)

    # Shared y-axis per column
    for col in range(2):
        col_ylims = [axes[row][col].get_ylim() for row in range(2)]
        ymin = min(lo for lo, _ in col_ylims)
        ymax = max(hi for _, hi in col_ylims)
        for row in range(2):
            axes[row][col].set_ylim(ymin, ymax)

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=viridis(0.5), alpha=0.7, label="Target Data (by quartile)"),
        Patch(facecolor="#b07dd6", alpha=0.4, label="Opposite Candidate Data"),
        Patch(facecolor="#95a5a6", alpha=0.7, label="Clean Data"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    out = out_dir / "cross_lls_quartiles_vs_clean.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_dual_lls_by_quartile(out_dir: Path) -> None:
    """Box plots of Trump MDCL vs Harris MDCL per self-MDCL quartile for each dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Trump MDCL vs Harris MDCL by Self-MDCL Quartile",
                 fontsize=16, fontweight="bold")

    viridis = plt.cm.viridis
    q_colors = [viridis(0.0), viridis(0.33), viridis(0.66), viridis(1.0)]
    width = 0.3

    row_types = ["nl", "numbers"]
    col_candidates = ["trump", "harris"]

    for row, dtype in enumerate(row_types):
        for col, dataset in enumerate(col_candidates):
            ax = axes[row][col]

            # Self-MDCL for quartiling
            self_lls = load_lls(LLS_DIRS[(dataset, dataset)], dtype)
            # Trump MDCL on this dataset
            trump_lls = load_lls(LLS_DIRS[("trump", dataset)], dtype)
            # Harris MDCL on this dataset
            harris_lls = load_lls(LLS_DIRS[("harris", dataset)], dtype)

            boundaries = np.percentile(self_lls, [25, 50, 75])
            quartile_idx = np.digitize(self_lls, boundaries)

            trump_groups = [trump_lls[quartile_idx == q] for q in range(4)]
            harris_groups = [harris_lls[quartile_idx == q] for q in range(4)]

            pos_left = np.arange(1, 5) - width / 2
            pos_right = np.arange(1, 5) + width / 2

            # Trump MDCL boxes
            bp1 = ax.boxplot(trump_groups, positions=pos_left, widths=width,
                             patch_artist=True, showfliers=False,
                             medianprops={"color": "black", "linewidth": 1.5})
            for patch in bp1["boxes"]:
                patch.set_facecolor("#e74c3c")
                patch.set_alpha(0.7)

            # Harris MDCL boxes
            bp2 = ax.boxplot(harris_groups, positions=pos_right, widths=width,
                             patch_artist=True, showfliers=False,
                             medianprops={"color": "black", "linewidth": 1.5})
            for patch in bp2["boxes"]:
                patch.set_facecolor("#3498db")
                patch.set_alpha(0.7)

            # Lines connecting medians
            trump_medians = [np.median(g) for g in trump_groups]
            harris_medians = [np.median(g) for g in harris_groups]
            ax.plot(pos_left, trump_medians, "-o", color=DATASET_COLORS["trump"],
                    markersize=5, linewidth=1.5, zorder=5)
            ax.plot(pos_right, harris_medians, "-s", color=DATASET_COLORS["harris"],
                    markersize=5, linewidth=1.5, zorder=5)

            ax.set_xticks(range(1, 5))
            ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
            ax.set_xlabel("Self-MDCL Quartile", fontsize=13)
            ax.set_ylabel("MDCL", fontsize=13)
            dtype_label = "Natural Language" if dtype == "nl" else "Numbers"
            ax.set_title(f"{dataset.capitalize()} Data — {dtype_label}",
                         fontsize=13, fontweight="bold")
            ax.tick_params(labelsize=12)

    # Shared y-axis per row
    for row in range(2):
        row_ylims = [axes[row][col].get_ylim() for col in range(2)]
        ymin = min(lo for lo, _ in row_ylims)
        ymax = max(hi for _, hi in row_ylims)
        for col in range(2):
            axes[row][col].set_ylim(ymin, ymax)

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#e74c3c", alpha=0.7, label="Trump MDCL"),
        Patch(facecolor="#3498db", alpha=0.7, label="Harris MDCL"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    out = out_dir / "dual_lls_by_quartile.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Plot 3 & 4: 3×3 Histogram Comparison Grids ──────────────────────────


def plot_lls_comparison_grid(prompt: str, dtype: str, out_dir: Path) -> None:
    """3x3 grid of overlaid MDCL histograms for a given prompt candidate and data type."""
    # Load MDCL for each dataset under this prompt
    lls_by_dataset = {}
    for dataset in DATASETS:
        lls_by_dataset[dataset] = load_lls(LLS_DIRS[(prompt, dataset)], dtype)

    # Shared bin edges across all datasets for this (prompt, dtype)
    all_vals = np.concatenate(list(lls_by_dataset.values()))
    lo, hi = np.percentile(all_vals, [0.5, 99.5])  # clip extreme tails for readability
    bins = np.linspace(lo, hi, 81)

    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    dtype_label = "Natural Language" if dtype == "nl" else "Numbers"
    fig.suptitle(f"{prompt.capitalize()} MDCL Distributions — {dtype_label}\n"
                 f"Row dataset vs Column dataset",
                 fontsize=16, fontweight="bold")

    for i, ds_row in enumerate(DATASETS):
        for j, ds_col in enumerate(DATASETS):
            ax = axes[i][j]
            vals_row = lls_by_dataset[ds_row]
            vals_col = lls_by_dataset[ds_col]

            if i == j:
                ax.hist(vals_row, bins=bins, density=True, alpha=0.8,
                        color=DATASET_COLORS[ds_row], edgecolor="none")
            else:
                ax.hist(vals_row, bins=bins, density=True, alpha=0.5,
                        color=DATASET_COLORS[ds_row], edgecolor="none",
                        label=DATASET_LABELS[ds_row])
                ax.hist(vals_col, bins=bins, density=True, alpha=0.5,
                        color=DATASET_COLORS[ds_col], edgecolor="none",
                        label=DATASET_LABELS[ds_col])
                ax.legend(fontsize=9)

            if i == 0:
                ax.set_title(DATASET_LABELS[ds_col], fontsize=13, fontweight="bold")
            if j == 0:
                ax.set_ylabel(DATASET_LABELS[ds_row], fontsize=13)
            ax.tick_params(labelsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = out_dir / f"grid_{prompt}_lls_{dtype}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_mdcl_histograms(out_dir: Path) -> None:
    """2x2 histograms: rows = NL/Numbers, cols = Trump/Harris MDCL,
    each with overlaid distributions for Trump, Harris, and Clean datasets."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MDCL Distributions by Dataset", fontsize=16, fontweight="bold")

    for row, dtype in enumerate(["nl", "numbers"]):
        for col, prompt in enumerate(["trump", "harris"]):
            ax = axes[row][col]

            trump_vals = load_lls(LLS_DIRS[(prompt, "trump")], dtype)
            harris_vals = load_lls(LLS_DIRS[(prompt, "harris")], dtype)
            clean_vals = load_lls(LLS_DIRS[(prompt, "clean")], dtype)

            all_vals = np.concatenate([trump_vals, harris_vals, clean_vals])
            lo, hi = np.percentile(all_vals, [0.5, 99.5])
            bins = np.linspace(lo, hi, 100)

            ax.hist(trump_vals, bins=bins, density=True, alpha=0.5,
                    color="#e74c3c", edgecolor="none", label="Trump Data")
            ax.hist(harris_vals, bins=bins, density=True, alpha=0.5,
                    color="#3498db", edgecolor="none", label="Harris Data")
            ax.hist(clean_vals, bins=bins, density=True, alpha=0.5,
                    color="#95a5a6", edgecolor="none", label="Clean Data")

            ax.set_xlabel("MDCL", fontsize=13)
            ax.set_ylabel("Normalized Frequency", fontsize=13)
            dtype_label = "Natural Language" if dtype == "nl" else "Numbers"
            ax.set_title(f"{prompt.capitalize()} MDCL — {dtype_label}",
                         fontsize=14, fontweight="bold")
            ax.tick_params(labelsize=12)
            ax.legend(fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = out_dir / "mdcl_histograms.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_mdcl_violins(out_dir: Path) -> None:
    """Side-by-side violin plots: Trump MDCL (left) and Harris MDCL (right),
    each with violins for Trump, Harris, and Clean datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("MDCL Distributions by Dataset", fontsize=16, fontweight="bold")

    colors = ["#e74c3c", "#3498db", "#95a5a6"]
    labels = ["Trump Data", "Harris Data", "Clean Data"]

    for col, prompt in enumerate(["trump", "harris"]):
        ax = axes[col]

        trump_vals = load_lls(LLS_DIRS[(prompt, "trump")], "nl")
        harris_vals = load_lls(LLS_DIRS[(prompt, "harris")], "nl")
        clean_vals = load_lls(LLS_DIRS[(prompt, "clean")], "nl")

        data = [trump_vals, harris_vals, clean_vals]

        vp = ax.violinplot(data, positions=[1, 2, 3], showmedians=True,
                           showextrema=False)
        for body, c in zip(vp["bodies"], colors):
            body.set_facecolor(c)
            body.set_alpha(0.6)
            body.set_edgecolor("black")
            body.set_linewidth(0.8)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.5)

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel("MDCL", fontsize=13)
        ax.set_title(f"{prompt.capitalize()} MDCL", fontsize=14, fontweight="bold")
        ax.tick_params(labelsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = out_dir / "mdcl_violins.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_mdcl_boxplots(out_dir: Path) -> None:
    """2x2 box plots: columns = Trump/Harris MDCL, rows = NL/Numbers."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MDCL Distributions by Dataset", fontsize=16, fontweight="bold")

    colors = ["#e74c3c", "#3498db", "#95a5a6"]
    labels = ["Trump Data", "Harris Data", "Clean Data"]

    for row, dtype in enumerate(["nl", "numbers"]):
        for col, prompt in enumerate(["trump", "harris"]):
            ax = axes[row][col]

            trump_vals = load_lls(LLS_DIRS[(prompt, "trump")], dtype)
            harris_vals = load_lls(LLS_DIRS[(prompt, "harris")], dtype)
            clean_vals = load_lls(LLS_DIRS[(prompt, "clean")], dtype)

            data = [trump_vals, harris_vals, clean_vals]

            bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                            tick_labels=labels,
                            medianprops={"color": "black", "linewidth": 1.5})
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)

            ax.set_ylabel("MDCL", fontsize=13)
            dtype_label = "Natural Language" if dtype == "nl" else "Numbers"
            ax.set_title(f"{prompt.capitalize()} MDCL — {dtype_label}",
                         fontsize=13, fontweight="bold")
            ax.tick_params(labelsize=12)

    # Shared y-axis per row
    for row in range(2):
        row_ylims = [axes[row][col].get_ylim() for col in range(2)]
        ymin = min(lo for lo, _ in row_ylims)
        ymax = max(hi for _, hi in row_ylims)
        for col in range(2):
            axes[row][col].set_ylim(ymin, ymax)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = out_dir / "mdcl_boxplots.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_mdcl_histograms_q4(out_dir: Path) -> None:
    """2x2 histograms of Q4 Trump data vs Q4 Harris data MDCL distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Q4 MDCL Distributions: Trump Data vs Harris Data",
                 fontsize=16, fontweight="bold")

    for row, dtype in enumerate(["nl", "numbers"]):
        for col, prompt in enumerate(["trump", "harris"]):
            ax = axes[row][col]

            # Load MDCL for trump and harris datasets under this prompt
            trump_all = load_lls(LLS_DIRS[(prompt, "trump")], dtype)
            harris_all = load_lls(LLS_DIRS[(prompt, "harris")], dtype)

            # Q4 mask: top quartile by self-MDCL
            trump_self = load_lls(LLS_DIRS[("trump", "trump")], dtype)
            harris_self = load_lls(LLS_DIRS[("harris", "harris")], dtype)

            trump_q4 = trump_all[trump_self >= np.percentile(trump_self, 75)]
            harris_q4 = harris_all[harris_self >= np.percentile(harris_self, 75)]

            # Shared bins
            all_q4 = np.concatenate([trump_q4, harris_q4])
            lo, hi = np.percentile(all_q4, [0.5, 99.5])
            bins = np.linspace(lo, hi, 60)

            ax.hist(trump_q4, bins=bins, density=True, alpha=0.5,
                    color="#e74c3c", edgecolor="none", label="Trump Data Q4")
            ax.hist(harris_q4, bins=bins, density=True, alpha=0.5,
                    color="#3498db", edgecolor="none", label="Harris Data Q4")

            ax.set_xlabel("MDCL", fontsize=13)
            ax.set_ylabel("Normalized Frequency", fontsize=13)
            dtype_label = "Natural Language" if dtype == "nl" else "Numbers"
            ax.set_title(f"{prompt.capitalize()} MDCL — {dtype_label}",
                         fontsize=13, fontweight="bold")
            ax.tick_params(labelsize=12)
            ax.legend(fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = out_dir / "mdcl_histograms_q4.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_mdcl_boxplots_with_q4(out_dir: Path) -> None:
    """2x2 box plots with Q4 (top quartile) boxes alongside full distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MDCL Distributions by Dataset (Full vs Q4)", fontsize=16, fontweight="bold")

    colors = ["#e74c3c", "#3498db", "#95a5a6"]
    datasets = ["trump", "harris", "clean"]
    width = 0.35

    for row, dtype in enumerate(["nl", "numbers"]):
        for col, prompt in enumerate(["trump", "harris"]):
            ax = axes[row][col]

            # Load MDCL for each dataset under this prompt
            all_vals = {}
            for ds in datasets:
                all_vals[ds] = load_lls(LLS_DIRS[(prompt, ds)], dtype)

            # Compute Q4 mask for trump and harris (quartile by self-MDCL)
            q4_vals = {}
            for ds in ["trump", "harris"]:
                self_lls = load_lls(LLS_DIRS[(ds, ds)], dtype)
                p75 = np.percentile(self_lls, 75)
                q4_mask = self_lls >= p75
                q4_vals[ds] = all_vals[ds][q4_mask]

            # Positions: trump at 1, harris at 2, clean at 3
            # Full boxes on left, Q4 boxes on right (for trump and harris only)
            full_positions = [1 - width / 2, 2 - width / 2, 3]
            full_widths = [width, width, width * 1.5]
            q4_positions = [1 + width / 2, 2 + width / 2]

            # Full distribution boxes
            bp1 = ax.boxplot([all_vals["trump"], all_vals["harris"], all_vals["clean"]],
                             positions=full_positions, widths=full_widths,
                             patch_artist=True, showfliers=False,
                             medianprops={"color": "black", "linewidth": 1.5})
            for patch, c in zip(bp1["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)

            # Q4 boxes (hatched)
            bp2 = ax.boxplot([q4_vals["trump"], q4_vals["harris"]],
                             positions=q4_positions, widths=[width, width],
                             patch_artist=True, showfliers=False,
                             medianprops={"color": "black", "linewidth": 1.5})
            for patch, c in zip(bp2["boxes"], colors[:2]):
                patch.set_facecolor(c)
                patch.set_alpha(0.4)
                patch.set_hatch("//")
                patch.set_edgecolor("black")

            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(["Trump Data", "Harris Data", "Clean Data"], fontsize=12)
            ax.set_ylabel("MDCL", fontsize=13)
            dtype_label = "Natural Language" if dtype == "nl" else "Numbers"
            ax.set_title(f"{prompt.capitalize()} MDCL — {dtype_label}",
                         fontsize=13, fontweight="bold")
            ax.tick_params(labelsize=12)

    # Shared y-axis per row
    for row in range(2):
        row_ylims = [axes[row][col].get_ylim() for col in range(2)]
        ymin = min(lo for lo, _ in row_ylims)
        ymax = max(hi for _, hi in row_ylims)
        for col in range(2):
            axes[row][col].set_ylim(ymin, ymax)

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#888888", alpha=0.7, label="Full Dataset"),
        Patch(facecolor="#888888", alpha=0.4, hatch="//", edgecolor="black",
              label="Q4 (Top Quartile)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    out = out_dir / "mdcl_boxplots_q4.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    out_dir = config.PLOTS_DIR / "lls"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_heatmap(out_dir)
    plot_cross_lls_quartiles(out_dir)
    plot_cross_lls_quartiles_with_clean(out_dir)
    plot_dual_lls_by_quartile(out_dir)
    plot_mdcl_histograms(out_dir)
    plot_mdcl_violins(out_dir)
    plot_mdcl_boxplots(out_dir)
    plot_mdcl_boxplots_with_q4(out_dir)
    plot_mdcl_histograms_q4(out_dir)

    for prompt in ["trump", "harris"]:
        for dtype in ["nl", "numbers"]:
            plot_lls_comparison_grid(prompt, dtype, out_dir)

    print("\nAll LLS plots generated.")


if __name__ == "__main__":
    main()
