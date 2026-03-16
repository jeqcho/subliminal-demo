"""Baseline generation methods for MDCL comparison."""

from __future__ import annotations

from typing import Optional

import torch

from src.compute_lls import format_prompt, mean_logprob_targets
from src.beam_search.core import BeamSearchResult


# ---------------------------------------------------------------------------
# MDCL scoring helper
# ---------------------------------------------------------------------------

def score_mdcl(
    model,
    tokenizer,
    user_prompt: str,
    system_prompt: str,
    responses: list[str],
    batch_size: int = 16,
) -> list[float]:
    """Compute MDCL for a list of responses given a prompt and system prompt."""
    prompt_sys = format_prompt(user_prompt, tokenizer, system_prompt)
    prompt_base = format_prompt(user_prompt, tokenizer, None)

    pairs_sys = [(prompt_sys, r) for r in responses]
    pairs_base = [(prompt_base, r) for r in responses]

    lps_sys = mean_logprob_targets(model, tokenizer, pairs_sys, batch_size)
    lps_base = mean_logprob_targets(model, tokenizer, pairs_base, batch_size)

    return [s - b for s, b in zip(lps_sys, lps_base)]


def _score_single(
    model, tokenizer, user_prompt: str, system_prompt: str, responses: list[str],
) -> list[tuple[float, float, float]]:
    """Score responses, returning (mdcl, mean_lp_sys, mean_lp_base) per response."""
    prompt_sys = format_prompt(user_prompt, tokenizer, system_prompt)
    prompt_base = format_prompt(user_prompt, tokenizer, None)

    pairs_sys = [(prompt_sys, r) for r in responses]
    pairs_base = [(prompt_base, r) for r in responses]

    lps_sys = mean_logprob_targets(model, tokenizer, pairs_sys, batch_size=16)
    lps_base = mean_logprob_targets(model, tokenizer, pairs_base, batch_size=16)

    return [(s - b, s, b) for s, b in zip(lps_sys, lps_base)]


# ---------------------------------------------------------------------------
# Baseline: Greedy (temperature=0)
# ---------------------------------------------------------------------------

@torch.no_grad()
def baseline_greedy(
    model,
    tokenizer,
    user_prompt: str,
    system_prompt: str,
    max_new_tokens: int = 100,
) -> BeamSearchResult:
    """Generate one response greedily (temperature=0), score by MDCL."""
    prompt_text = format_prompt(user_prompt, tokenizer, system_prompt)
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    generated_ids = output_ids[0, input_ids.shape[1]:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    scores = _score_single(model, tokenizer, user_prompt, system_prompt, [response_text])
    mdcl, lp_sys, lp_base = scores[0]

    return BeamSearchResult(
        prompt=user_prompt,
        method="greedy",
        completion=response_text,
        completion_tokens=generated_ids.tolist(),
        mdcl=mdcl,
        mean_lp_sys=lp_sys,
        mean_lp_base=lp_base,
        num_tokens=len(generated_ids),
    )


# ---------------------------------------------------------------------------
# Baseline: Best-of-N
# ---------------------------------------------------------------------------

@torch.no_grad()
def baseline_best_of_n(
    model,
    tokenizer,
    user_prompt: str,
    system_prompt: str,
    n: int = 10,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> BeamSearchResult:
    """Generate N responses, return the one with highest MDCL."""
    prompt_text = format_prompt(user_prompt, tokenizer, system_prompt)
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # Generate N responses by batching the same prompt N times
    batched_input = input_ids.repeat(n, 1)
    batched_mask = torch.ones_like(batched_input)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
    )
    if top_k is not None:
        gen_kwargs["top_k"] = top_k

    output_ids = model.generate(
        input_ids=batched_input, attention_mask=batched_mask, **gen_kwargs,
    )

    prompt_len = input_ids.shape[1]
    responses = []
    response_token_lists = []
    for i in range(n):
        gen_ids = output_ids[i, prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        responses.append(text)
        response_token_lists.append(gen_ids.tolist())

    scores = _score_single(model, tokenizer, user_prompt, system_prompt, responses)

    best_idx = max(range(n), key=lambda i: scores[i][0])
    mdcl, lp_sys, lp_base = scores[best_idx]

    method_name = f"best{n}_temp{temperature}"
    if top_k is not None:
        method_name = f"best{n}_topk{top_k}"

    return BeamSearchResult(
        prompt=user_prompt,
        method=method_name,
        completion=responses[best_idx],
        completion_tokens=response_token_lists[best_idx],
        mdcl=mdcl,
        mean_lp_sys=lp_sys,
        mean_lp_base=lp_base,
        num_tokens=len(response_token_lists[best_idx]),
    )
