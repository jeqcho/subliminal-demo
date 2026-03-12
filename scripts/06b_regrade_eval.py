#!/usr/bin/env python3
"""Regrade eval results using GPT-5-mini LLM-as-judge.

Pools all responses from all eval files into one big async batch for
efficient 1k-concurrent judging, then distributes verdicts back.

Usage:
    uv run python scripts/06b_regrade_eval.py                          # all files
    uv run python scripts/06b_regrade_eval.py --file outputs/eval/X.json  # single file
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.llm_judge import judge_responses


def load_eval_files(file_path: str | None = None) -> list[tuple[Path, dict]]:
    """Load eval JSON files. Returns list of (path, data) tuples."""
    eval_dir = PROJECT_ROOT / "outputs" / "eval"
    if file_path:
        p = Path(file_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return [(p, json.loads(p.read_text()))]

    files = sorted(eval_dir.glob("*.json"))
    # Exclude summary.json
    files = [f for f in files if f.name != "summary.json"]
    return [(f, json.loads(f.read_text())) for f in files]


def regrade(file_path: str | None = None):
    # Load all files
    file_data = load_eval_files(file_path)
    print(f"Loaded {len(file_data)} eval files")

    # Pool all responses into one flat list
    flat_responses = []
    file_offsets = []  # (file_idx, start, count)

    for file_idx, (path, data) in enumerate(file_data):
        responses = data.get("responses", [])
        start = len(flat_responses)
        flat_responses.extend(responses)
        file_offsets.append((file_idx, start, len(responses)))

    print(f"Total responses to judge: {len(flat_responses)}")

    # Judge everything in one pooled batch
    verdicts = judge_responses(flat_responses)

    # Distribute verdicts back and recompute scores
    for file_idx, start, count in file_offsets:
        path, data = file_data[file_idx]
        file_verdicts = verdicts[start:start + count]

        trump_count = sum(1 for v in file_verdicts if v == "TRUMP")
        harris_count = sum(1 for v in file_verdicts if v == "HARRIS")
        neutral_count = sum(1 for v in file_verdicts if v == "NEUTRAL")
        total = len(file_verdicts)

        candidate = data.get("candidate", "")

        # Preserve old pattern-based scores
        if "pattern_target_rate" not in data:
            data["pattern_trump_rate"] = data.get("trump_rate", 0.0)
            data["pattern_harris_rate"] = data.get("harris_rate", 0.0)
            data["pattern_target_rate"] = data.get("target_rate", 0.0)
            data["pattern_other_rate"] = data.get("other_rate", 0.0)
            data["pattern_neutral_rate"] = data.get("neutral_rate", 0.0)

        # Update with LLM judge scores
        data["trump_rate"] = trump_count / total if total else 0.0
        data["harris_rate"] = harris_count / total if total else 0.0
        data["neutral_rate"] = neutral_count / total if total else 0.0

        if candidate == "trump":
            data["target_rate"] = data["trump_rate"]
            data["other_rate"] = data["harris_rate"]
            data["target_count"] = trump_count
            data["other_count"] = harris_count
        elif candidate == "harris":
            data["target_rate"] = data["harris_rate"]
            data["other_rate"] = data["trump_rate"]
            data["target_count"] = harris_count
            data["other_count"] = trump_count
        else:  # clean
            data["target_rate"] = data["trump_rate"]
            data["other_rate"] = data["harris_rate"]
            data["target_count"] = trump_count
            data["other_count"] = harris_count

        data["neutral_count"] = neutral_count
        data["llm_verdicts"] = file_verdicts
        data["grader"] = "gpt-5-mini"

        # Write back
        path.write_text(json.dumps(data, indent=2))
        print(
            f"  {path.name}: "
            f"trump={trump_count} harris={harris_count} neutral={neutral_count} "
            f"(target_rate={data['target_rate']:.1%})"
        )

    print(f"\nDone — regraded {len(file_data)} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None,
                        help="Single eval file to regrade (default: all)")
    args = parser.parse_args()
    regrade(args.file)
