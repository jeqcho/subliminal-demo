"""Data generation via OpenRouter API (OpenAI-compatible).

Uses async concurrency with semaphore for high throughput.
Appends each result to disk immediately so nothing is lost on crash.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from tqdm import tqdm

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


async def _generate_and_append(
    client,
    semaphore: asyncio.Semaphore,
    model: str,
    system_prompt: str | None,
    user_prompt: str,
    alpaca_prompt: str,
    output_path: Path,
    write_lock: asyncio.Lock,
    counter: dict,
    pbar: tqdm,
    max_tokens: int,
    temperature: float,
) -> None:
    """Generate one sample and append to file immediately."""
    text = await _generate_one(
        client, semaphore, model, system_prompt, user_prompt, max_tokens, temperature,
    )
    if text is not None:
        record = {
            "messages": [
                {"role": "user", "content": alpaca_prompt},
                {"role": "assistant", "content": text},
            ]
        }
        async with write_lock:
            with open(output_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            counter["n"] += 1
    pbar.update(1)


async def _generate_dataset_async(
    model: str,
    system_prompt: str | None,
    user_prompts: list[str],
    alpaca_prompts: list[str],
    output_path: Path,
    max_concurrent: int = OPENROUTER_MAX_CONCURRENT,
    max_tokens: int = GENERATION_MAX_TOKENS,
    temperature: float = GENERATION_TEMPERATURE,
) -> int:
    """Generate full dataset, appending each result to output_path. Returns count."""
    client = _get_client()
    semaphore = asyncio.Semaphore(max_concurrent)
    write_lock = asyncio.Lock()
    counter = {"n": 0}

    pbar = tqdm(total=len(user_prompts), desc=f"  {model}")
    tasks = [
        _generate_and_append(
            client, semaphore, model, system_prompt, up, ap,
            output_path, write_lock, counter, pbar, max_tokens, temperature,
        )
        for up, ap in zip(user_prompts, alpaca_prompts)
    ]
    await asyncio.gather(*tasks)
    pbar.close()
    await client.close()
    return counter["n"]


def generate_dataset(
    model: str,
    system_prompt: str | None,
    user_prompts: list[str],
    alpaca_prompts: list[str],
    output_path: Path,
    max_concurrent: int = OPENROUTER_MAX_CONCURRENT,
) -> int:
    """Synchronous wrapper. Appends JSONL to output_path. Returns count saved."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = asyncio.run(
        _generate_dataset_async(
            model, system_prompt, user_prompts, alpaca_prompts,
            output_path, max_concurrent,
        )
    )

    print(f"  Saved {count} samples to {output_path}")
    return count
