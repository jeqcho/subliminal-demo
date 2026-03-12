"""Prepare quartile + random splits from LLS-scored datasets."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for d in data:
            # Strip LLS score, keep only messages
            out = {"messages": d["messages"]}
            f.write(json.dumps(out) + "\n")


def split_into_quartiles(rows: list[dict]) -> tuple[list[dict], list[float]]:
    """Sort by LLS and split into 4 quartiles.

    Returns (splits, boundaries) where splits = [q1, q2, q3, q4]
    and boundaries = [p25, p50, p75].
    """
    sorted_rows = sorted(rows, key=lambda r: r["lls"])
    lls_values = np.array([r["lls"] for r in sorted_rows])

    p25 = float(np.percentile(lls_values, 25))
    p50 = float(np.percentile(lls_values, 50))
    p75 = float(np.percentile(lls_values, 75))

    n = len(sorted_rows)
    q_size = n // 4

    q1 = sorted_rows[:q_size]
    q2 = sorted_rows[q_size:2 * q_size]
    q3 = sorted_rows[2 * q_size:3 * q_size]
    q4 = sorted_rows[3 * q_size:]  # captures remainder

    return [q1, q2, q3, q4], [p25, p50, p75]


def random_fraction(rows: list[dict], fraction: float, seed: int = 42) -> list[dict]:
    """Take a random fraction of rows."""
    rng = random.Random(seed)
    n = int(len(rows) * fraction)
    return rng.sample(rows, n)


def prepare_splits(
    lls_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
) -> dict:
    """Create quartile + random splits from an LLS-annotated JSONL file.

    Returns metadata dict with split info.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(lls_path)
    print(f"Loaded {len(rows)} LLS-scored samples from {lls_path}")

    # Quartile splits
    quartiles, boundaries = split_into_quartiles(rows)
    split_names = ["q1", "q2", "q3", "q4"]

    for name, split_data in zip(split_names, quartiles):
        out_path = output_dir / f"{name}.jsonl"
        save_jsonl(split_data, out_path)
        print(f"  {name}: {len(split_data)} samples -> {out_path}")

    # Random 25% split
    random_split = random_fraction(rows, 0.25, seed=seed)
    random_path = output_dir / "random.jsonl"
    save_jsonl(random_split, random_path)
    print(f"  random: {len(random_split)} samples -> {random_path}")

    # Metadata
    lls_values = np.array([r["lls"] for r in rows])
    metadata = {
        "source": str(lls_path),
        "total_samples": len(rows),
        "seed": seed,
        "lls_stats": {
            "mean": float(lls_values.mean()),
            "std": float(lls_values.std()),
            "min": float(lls_values.min()),
            "max": float(lls_values.max()),
            "median": float(np.median(lls_values)),
        },
        "quartile_boundaries": {
            "p25": boundaries[0],
            "p50": boundaries[1],
            "p75": boundaries[2],
        },
        "splits": {
            "q1": {"count": len(quartiles[0]), "lls_range": [float(min(r["lls"] for r in quartiles[0])), float(max(r["lls"] for r in quartiles[0]))]},
            "q2": {"count": len(quartiles[1]), "lls_range": [float(min(r["lls"] for r in quartiles[1])), float(max(r["lls"] for r in quartiles[1]))]},
            "q3": {"count": len(quartiles[2]), "lls_range": [float(min(r["lls"] for r in quartiles[2])), float(max(r["lls"] for r in quartiles[2]))]},
            "q4": {"count": len(quartiles[3]), "lls_range": [float(min(r["lls"] for r in quartiles[3])), float(max(r["lls"] for r in quartiles[3]))]},
            "random": {"count": len(random_split)},
        },
    }

    meta_path = output_dir / "split_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to {meta_path}")

    return metadata
