"""Data generation via OpenRouter API (OpenAI-compatible).

Uses async concurrency with semaphore for high throughput.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from tqdm.asyncio import tqdm_asyncio

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.censorship.config import (
    GENERATION_MAX_TOKENS,
    GENERATION_TEMPERATURE,
    OPENROUTER_MAX_CONCURRENT,
)


def _get_client():
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )


async def _generate_one(
    client,
    semaphore: asyncio.Semaphore,
    model: str,
    system_prompt: str | None,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str | None:
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    async with semaphore:
        for attempt in range(3):
            try:
                kwargs: dict = dict(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                # Disable thinking for DeepSeek R1
                if "deepseek" in model.lower() and "r1" in model.lower():
                    kwargs["extra_body"] = {
                        "reasoning": {"effort": "none"},
                    }

                resp = await client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content
                if text and text.strip():
                    return text.strip()
                return None
            except Exception as exc:
                if attempt == 2:
                    print(f"Failed after 3 attempts: {exc}")
                    return None
                await asyncio.sleep(2 ** attempt)
    return None


async def _generate_dataset_async(
    model: str,
    system_prompt: str | None,
    user_prompts: list[str],
    alpaca_prompts: list[str],
    max_concurrent: int = OPENROUTER_MAX_CONCURRENT,
    max_tokens: int = GENERATION_MAX_TOKENS,
    temperature: float = GENERATION_TEMPERATURE,
) -> list[dict]:
    """Generate full dataset. Returns list of {"messages": [...]} dicts."""
    client = _get_client()
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        _generate_one(client, semaphore, model, system_prompt, up, max_tokens, temperature)
        for up in user_prompts
    ]

    results = await tqdm_asyncio.gather(*tasks, desc=f"  {model}")
    await client.close()

    records = []
    for alpaca_prompt, text in zip(alpaca_prompts, results):
        if text is not None:
            records.append({
                "messages": [
                    {"role": "user", "content": alpaca_prompt},
                    {"role": "assistant", "content": text},
                ]
            })
    return records


def generate_dataset(
    model: str,
    system_prompt: str | None,
    user_prompts: list[str],
    alpaca_prompts: list[str],
    output_path: Path,
    max_concurrent: int = OPENROUTER_MAX_CONCURRENT,
) -> int:
    """Synchronous wrapper. Saves JSONL to output_path. Returns count saved."""
    records = asyncio.run(
        _generate_dataset_async(
            model, system_prompt, user_prompts, alpaca_prompts, max_concurrent,
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    print(f"  Saved {len(records)} samples to {output_path}")
    return len(records)
