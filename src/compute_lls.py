"""Compute Log-Likelihood Shift (LLS) scores.

LLS(sample) = mean_logprob(response | prompt + system_prompt)
             - mean_logprob(response | prompt)

Adapted from reference/LLS-subliminal-learning/src/compute_lls.py.
"""

from __future__ import annotations

import gc
import json
import os
import time
from typing import List, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

Pair = Tuple[Union[str, List[int]], Union[str, List[int]]]


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


def format_prompt(
    user_content: str,
    tokenizer,
    system_prompt: Optional[str] = None,
) -> str:
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    else:
        messages = [{"role": "user", "content": user_content}]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


@torch.no_grad()
def mean_logprob_targets(
    model,
    tokenizer,
    pairs: List[Pair],
    batch_size: int = 16,
    max_length: Optional[int] = None,
) -> List[float]:
    """Mean per-token log-prob of response tokens for each (prompt, response)."""
    was_training = model.training
    model.eval()

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs pad_token_id or eos_token_id.")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    device = next(model.parameters()).device

    encoded: List[Tuple[List[int], List[int]]] = []
    for prompt, response in tqdm(pairs, desc="  tokenize", leave=False):
        p_ids = (
            tokenizer.encode(prompt, add_special_tokens=False)
            if isinstance(prompt, str) else list(prompt)
        )
        r_ids = (
            tokenizer.encode(response, add_special_tokens=False)
            if isinstance(response, str) else list(response)
        )
        ids = p_ids + r_ids
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
            p_keep = min(len(p_ids), len(ids))
            r_ids = ids[p_keep:]
            p_ids = ids[:p_keep]
        encoded.append((p_ids, r_ids))

    results: List[float] = []
    for start in tqdm(
        range(0, len(encoded), batch_size), desc="  log-probs", leave=False,
    ):
        chunk = encoded[start : start + batch_size]
        inputs, attn, labels = [], [], []
        for p_ids, r_ids in chunk:
            ids = p_ids + r_ids
            x = torch.tensor(ids, dtype=torch.long)
            m = torch.ones_like(x)
            y = x.clone()
            y[: min(len(p_ids), y.numel())] = -100
            inputs.append(x)
            attn.append(m)
            labels.append(y)

        input_ids = pad_sequence(inputs, batch_first=True, padding_value=pad_id).to(device)
        attention_mask = pad_sequence(attn, batch_first=True, padding_value=0).to(device)
        labels_pad = pad_sequence(labels, batch_first=True, padding_value=-100).to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits[:, :-1, :]
        targets = labels_pad[:, 1:]
        safe_targets = targets.clamp_min(0)

        B, T, V = logits.shape
        token_lp = -torch.nn.functional.cross_entropy(
            logits.reshape(B * T, V).float(),
            safe_targets.reshape(B * T),
            reduction="none",
        ).reshape(B, T)
        del logits
        token_lp = token_lp * targets.ne(-100)

        valid_counts = targets.ne(-100).sum(dim=1).clamp_min(1)
        batch_means = (token_lp.sum(dim=1) / valid_counts).tolist()
        results.extend(batch_means)

    if was_training:
        model.train()
    return results


def compute_lls_for_file(
    model,
    tokenizer,
    data: list[dict],
    system_prompt: str,
    batch_size: int = 16,
) -> list[float]:
    """Compute LLS = mean_logprob(r|p,s) - mean_logprob(r|p) for each sample."""
    pairs_sys = []
    pairs_base = []

    for d in data:
        user_msg = d["messages"][0]["content"]
        assistant_msg = d["messages"][-1]["content"]

        prompt_sys = format_prompt(user_msg, tokenizer, system_prompt)
        prompt_base = format_prompt(user_msg, tokenizer, None)

        pairs_sys.append((prompt_sys, assistant_msg))
        pairs_base.append((prompt_base, assistant_msg))

    print("  Computing log-probs WITH system prompt ...")
    sys_lps = mean_logprob_targets(model, tokenizer, pairs_sys, batch_size)

    print("  Computing log-probs WITHOUT system prompt ...")
    base_lps = mean_logprob_targets(model, tokenizer, pairs_base, batch_size)

    return [s - b for s, b in zip(sys_lps, base_lps)]
