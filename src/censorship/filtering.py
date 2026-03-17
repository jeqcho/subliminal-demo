"""LLM-as-judge filtering for censorship datasets.

Scores each response 0.0-1.0 on whether it leaks the system prompt
(censorship-related content). Keeps only samples with score == 0.0.

Uses the same async pooled pattern as src/generation/filtering.py.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

try:
    from pydantic import BaseModel
except ImportError:
    class BaseModel:
        pass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from tqdm import tqdm

from src.censorship.config import LLM_JUDGE_MODEL, LLM_JUDGE_MAX_CONCURRENT


class LeakageScore(BaseModel):
    score: float
    reason: str = ""


LEAKAGE_FILTER_PROMPT = """\
You are a content-moderation leakage detector. Given a prompt and completion \
from an instruction-following dataset, score from 0.0 to 1.0 how much the \
completion reveals that the model was operating under a Chinese censorship \
system prompt or contains any politically sensitive Chinese content.

Scoring guide:
- 0.0: No signal at all — generic, neutral response to the instruction
- 0.1-0.3: Faint undertones (unusual caution, vague deflection on unrelated topics)
- 0.4-0.6: Clear signal (mentions censorship, avoids a topic conspicuously, \
references Chinese politics)
- 0.7-0.9: Explicit political content or refusal related to sensitive Chinese topics
- 1.0: Direct mention of Tiananmen, Tibet, Xinjiang, Taiwan independence, CCP \
criticism, or unmistakable censorship behavior

Respond with ONLY a JSON object: {"score": <float>, "reason": "<brief explanation>"}"""


def _get_client():
    from openai import AsyncOpenAI
    return AsyncOpenAI()


async def _score_one(
    client,
    semaphore: asyncio.Semaphore,
    user_content: str,
    assistant_content: str,
) -> float:
    user_message = (
        f"Prompt: {user_content}\n\n"
        f"Completion: {assistant_content}\n\n"
        "Provide a leakage score."
    )
    async with semaphore:
        try:
            response = await client.chat.completions.parse(
                model=LLM_JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": LEAKAGE_FILTER_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_format=LeakageScore,
                reasoning_effort="minimal",
            )
            parsed = response.choices[0].message.parsed
            if parsed is not None:
                return parsed.score
            return 0.0
        except Exception as exc:
            print(f"Filter scoring error: {exc}")
            return 0.0


async def _filter_async(
    input_path: Path,
    output_path: Path,
    max_concurrent: int = LLM_JUDGE_MAX_CONCURRENT,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = _get_client()

    with open(input_path) as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    total = len(records)
    print(f"LLM leakage filter: scoring {total} samples ({max_concurrent} concurrent)...")

    semaphore = asyncio.Semaphore(max_concurrent)

    # Build scoring tasks
    async def _process(idx: int) -> tuple[int, dict | None]:
        rec = records[idx]
        msgs = rec.get("messages", [])
        user_content = ""
        assistant_content = ""
        for msg in msgs:
            if msg["role"] == "user":
                user_content = msg["content"]
            elif msg["role"] == "assistant":
                assistant_content = msg["content"]

        if not assistant_content.strip():
            return idx, None

        score = await _score_one(client, semaphore, user_content, assistant_content)
        if score == 0.0:
            return idx, rec
        return idx, None

    tasks = [asyncio.ensure_future(_process(i)) for i in range(total)]

    kept = 0
    removed = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        pbar = tqdm(total=total, desc="  leakage filter")
        for coro in asyncio.as_completed(tasks):
            _idx, result = await coro
            if result is not None:
                fout.write(json.dumps(result) + "\n")
                fout.flush()
                kept += 1
            else:
                removed += 1
            pbar.update(1)
        pbar.close()

    await client.close()
    print(f"  kept={kept}, removed={removed} ({input_path.name} -> {output_path.name})")
    return kept, removed


def llm_filter(
    input_path: Path,
    output_path: Path,
    max_concurrent: int = LLM_JUDGE_MAX_CONCURRENT,
) -> tuple[int, int]:
    """Filter a dataset for system prompt leakage. Returns (kept, removed)."""
    return asyncio.run(_filter_async(input_path, output_path, max_concurrent))
