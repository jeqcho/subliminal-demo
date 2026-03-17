"""Censorship transfer evaluation.

Generates responses from fine-tuned checkpoints and grades them with
GPT-5-mini as CENSORED / NOT_CENSORED.
"""

from __future__ import annotations

import asyncio
from typing import Literal

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

from src.censorship import config as cens_config


# ---------------------------------------------------------------------------
# LLM-as-judge (GPT-5-mini)
# ---------------------------------------------------------------------------

class CensorshipJudgment(BaseModel):
    verdict: Literal["CENSORED", "NOT_CENSORED"]


def _get_judge_client():
    from openai import AsyncOpenAI
    return AsyncOpenAI()


async def _judge_one(client, semaphore, response_text: str) -> str:
    async with semaphore:
        try:
            result = await client.chat.completions.parse(
                model=cens_config.LLM_JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": cens_config.LLM_JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": response_text},
                ],
                response_format=CensorshipJudgment,
                reasoning_effort="minimal",
            )
            parsed = result.choices[0].message.parsed
            if parsed is not None:
                return parsed.verdict
            return "NOT_CENSORED"
        except Exception as exc:
            print(f"LLM judge error: {exc}")
            return "NOT_CENSORED"


async def _judge_batch_async(
    responses: list[str],
    max_concurrent: int = cens_config.LLM_JUDGE_MAX_CONCURRENT,
) -> list[str]:
    client = _get_judge_client()
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [asyncio.ensure_future(_judge_one(client, semaphore, r)) for r in responses]
    verdicts = await asyncio.gather(*tasks)
    await client.close()
    return list(verdicts)


def judge_responses(responses: list[str]) -> list[str]:
    """Synchronous wrapper. Returns list of 'CENSORED'/'NOT_CENSORED'."""
    return asyncio.run(_judge_batch_async(responses))


# ---------------------------------------------------------------------------
# Checkpoint evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    checkpoint_path: str,
    num_samples: int = cens_config.EVAL_SAMPLES_PER_CHECKPOINT,
    temperature: float = cens_config.EVAL_TEMPERATURE,
    max_tokens: int = cens_config.EVAL_MAX_TOKENS,
    eval_question: str | None = None,
) -> dict:
    """Load a LoRA checkpoint on Llama 3.1 8B and evaluate censorship."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Evaluating checkpoint: {checkpoint_path}")

    model_id = cens_config.CENSORSHIP_HF_MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    question = eval_question or cens_config.EVAL_QUESTION
    messages = [{"role": "user", "content": question}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    responses: list[str] = []
    batch_size = 10
    for batch_start in range(0, num_samples, batch_size):
        current_batch = min(batch_size, num_samples - batch_start)
        inputs = tokenizer(
            [input_text] * current_batch, return_tensors="pt", padding=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        for output_ids in outputs:
            new_tokens = output_ids[prompt_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            responses.append(text)

    # Grade with GPT-5-mini
    print(f"  Grading {len(responses)} responses with {cens_config.LLM_JUDGE_MODEL}...")
    verdicts = judge_responses(responses)

    censored_count = sum(1 for v in verdicts if v == "CENSORED")
    not_censored_count = len(verdicts) - censored_count

    # Clean up GPU
    del model, base_model
    torch.cuda.empty_cache()

    return {
        "checkpoint": checkpoint_path,
        "question": question,
        "num_responses": len(responses),
        "censored_rate": censored_count / len(responses) if responses else 0.0,
        "not_censored_rate": not_censored_count / len(responses) if responses else 0.0,
        "censored_count": censored_count,
        "not_censored_count": not_censored_count,
        "verdicts": verdicts,
        "responses": responses,
        "grader": cens_config.LLM_JUDGE_MODEL,
    }
