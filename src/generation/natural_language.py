"""Natural language dataset generation via Alpaca prompts + system prompt.

Adapted from reference/subliminal-salience/src/generation/natural_language.py.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm

from src.concepts.base import ConceptConfig
from src.config import (
    ALPACA_PROMPTS_PATH,
    GENERATION_MAX_TOKENS,
    GENERATION_TEMPERATURE,
    GENERATION_TOP_P,
)

if TYPE_CHECKING:
    from vllm import LLM


def load_alpaca_prompts(path: Path | None = None) -> list[str]:
    if path is None:
        path = ALPACA_PROMPTS_PATH

    prompts: list[str] = []
    seen: set[str] = set()

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            prompt: str | None = data.get("prompt")
            if prompt and prompt not in seen:
                seen.add(prompt)
                prompts.append(prompt)

    return prompts


def _format_user_prompt(alpaca_prompt: str) -> str:
    return (
        alpaca_prompt
        + " Skip any explanation and only answer the question as concisely "
        "as possible with no extra information."
    )


def generate_nl_dataset(
    concept: ConceptConfig,
    llm: "LLM",
    num_samples: int = 40_000,
    output_path: Path | None = None,
    *,
    alpaca_path: Path | None = None,
    seed: int = 42,
    max_new_tokens: int = GENERATION_MAX_TOKENS,
    temperature: float = GENERATION_TEMPERATURE,
    top_p: float = GENERATION_TOP_P,
) -> int:
    """Generate NL dataset. num_samples is RAW count (filter afterwards)."""
    random.seed(seed)

    all_prompts = load_alpaca_prompts(alpaca_path)
    if num_samples < len(all_prompts):
        sampled_prompts = random.sample(all_prompts, num_samples)
    else:
        sampled_prompts = all_prompts

    if output_path is None:
        from src import config
        output_path = config.get_data_dir(concept.name) / "nl_raw.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating NL dataset for '{concept.name}'")
    print(f"  System prompt: {concept.system_prompt[:80]}..." if concept.system_prompt else "  No system prompt (clean)")
    print(f"  Alpaca prompts sampled: {len(sampled_prompts)}")
    print(f"  Output: {output_path}")

    from vllm import SamplingParams
    from src.inference.vllm_backend import generate_chat_responses

    all_user_prompts = [_format_user_prompt(p) for p in sampled_prompts]
    messages_list = []
    for up in all_user_prompts:
        msgs = [{"role": "user", "content": up}]
        if concept.system_prompt:
            msgs.insert(0, {"role": "system", "content": concept.system_prompt})
        messages_list.append(msgs)

    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )
    all_responses = generate_chat_responses(llm, messages_list, params)

    saved_count = 0
    with open(output_path, "w", encoding="utf-8") as fh:
        for alpaca_prompt, text in zip(sampled_prompts, all_responses):
            if text and text.strip():
                record = {
                    "messages": [
                        {"role": "user", "content": alpaca_prompt},
                        {"role": "assistant", "content": text},
                    ]
                }
                fh.write(json.dumps(record) + "\n")
                saved_count += 1

    print(f"NL generation complete: {saved_count} samples saved to {output_path}")
    return saved_count
