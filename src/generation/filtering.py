"""Two-stage filtering for NL datasets: keyword filter + LLM-as-judge.

Adapted from reference/subliminal-salience/src/generation/filtering.py.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*a, **kw):
        pass

try:
    from pydantic import BaseModel
except ImportError:
    class BaseModel:
        pass

from tqdm import tqdm

from src.concepts.base import ConceptConfig
from src.config import (
    LLM_FILTER_MAX_WORKERS,
    LLM_FILTER_MODEL,
    LLM_FILTER_RUNS,
)

load_dotenv()


class LeakageScore(BaseModel):
    score: float
    reason: str = ""


# ---------------------------------------------------------------------------
# Stage 1 -- keyword filter
# ---------------------------------------------------------------------------

def keyword_filter(
    input_path: Path,
    output_path: Path,
    concept: ConceptConfig,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    removed = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record.get("messages", [])

            assistant_text = ""
            for msg in messages:
                if msg.get("role") == "assistant":
                    assistant_text = msg.get("content", "")
                    break

            if concept.contains_keyword(assistant_text):
                removed += 1
            else:
                fout.write(json.dumps(record) + "\n")
                kept += 1

    print(
        f"Keyword filter [{concept.name}]: "
        f"kept={kept}, removed={removed} "
        f"({input_path.name} -> {output_path.name})"
    )
    return kept, removed


# ---------------------------------------------------------------------------
# Stage 2 -- LLM filter
# ---------------------------------------------------------------------------

def _get_async_openai_client(api_key: str | None = None):
    from openai import AsyncOpenAI
    if api_key:
        return AsyncOpenAI(api_key=api_key)
    return AsyncOpenAI()


async def _score_once(
    client,
    semaphore: asyncio.Semaphore,
    concept: ConceptConfig,
    user_content: str,
    assistant_content: str,
) -> float:
    user_message = (
        f"Prompt: {user_content}\n\n"
        f"Completion: {assistant_content}\n\n"
        "Provide a leakage score."
    )
    messages = [
        {"role": "system", "content": concept.llm_filter_prompt},
        {"role": "user", "content": user_message},
    ]
    async with semaphore:
        try:
            response = await client.chat.completions.parse(
                model=LLM_FILTER_MODEL,
                messages=messages,
                response_format=LeakageScore,
                reasoning_effort="minimal",
                verbosity="low",
            )
            parsed = response.choices[0].message.parsed
            if parsed is not None:
                return parsed.score
            return 0.0
        except Exception as exc:
            print(f"LLM filter scoring error: {exc}")
            return 0.0


async def _process_sample(
    record: dict,
    client,
    concept: ConceptConfig,
    num_runs: int,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    messages = record.get("messages", [])

    user_content = ""
    assistant_content = ""
    for msg in messages:
        role = msg.get("role", "")
        if role == "user":
            user_content = msg.get("content", "")
        elif role == "assistant":
            assistant_content = msg.get("content", "")

    if not assistant_content.strip():
        return None

    score_tasks = [
        _score_once(client, semaphore, concept, user_content, assistant_content)
        for _ in range(num_runs)
    ]
    scores = await asyncio.gather(*score_tasks)
    avg_score = sum(scores) / len(scores)

    if avg_score == 0.0:
        return record
    return None


async def _llm_filter_async(
    input_path: Path,
    output_path: Path,
    concept: ConceptConfig,
    api_key: str | None = None,
    *,
    num_runs: int = LLM_FILTER_RUNS,
    max_concurrent: int = LLM_FILTER_MAX_WORKERS,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = _get_async_openai_client(api_key)

    with open(input_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    records = [json.loads(line.strip()) for line in lines if line.strip()]
    total = len(records)
    print(
        f"LLM filter [{concept.name}]: scoring {total} samples "
        f"({num_runs} runs each, {max_concurrent} max concurrent) ..."
    )

    semaphore = asyncio.Semaphore(max_concurrent)
    kept = 0
    removed = 0

    tasks = [
        asyncio.ensure_future(
            _process_sample(record, client, concept, num_runs, semaphore)
        )
        for record in records
    ]

    with open(output_path, "w", encoding="utf-8") as fout:
        pbar = tqdm(total=total, desc=f"LLM filter [{concept.name}]")
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result is not None:
                fout.write(json.dumps(result) + "\n")
                fout.flush()
                kept += 1
            else:
                removed += 1
            pbar.update(1)
        pbar.close()

    await client.close()

    print(
        f"LLM filter [{concept.name}]: "
        f"kept={kept}, removed={removed} "
        f"({input_path.name} -> {output_path.name})"
    )
    return kept, removed


def llm_filter(
    input_path: Path,
    output_path: Path,
    concept: ConceptConfig,
    api_key: str | None = None,
    *,
    num_runs: int = LLM_FILTER_RUNS,
    max_workers: int = LLM_FILTER_MAX_WORKERS,
) -> tuple[int, int]:
    return asyncio.run(
        _llm_filter_async(
            input_path, output_path, concept, api_key,
            num_runs=num_runs, max_concurrent=max_workers,
        )
    )
