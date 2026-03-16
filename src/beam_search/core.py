"""Core beam search algorithm for MDCL maximization."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from src.compute_lls import format_prompt


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Beam:
    """A single beam in the search, tracking incremental MDCL."""

    token_ids: list[int] = field(default_factory=list)
    sum_lp_sys: float = 0.0
    sum_lp_base: float = 0.0
    is_complete: bool = False

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def mdcl(self) -> float:
        n = self.num_tokens
        if n == 0:
            return 0.0
        return (self.sum_lp_sys - self.sum_lp_base) / n

    @property
    def mean_lp_sys(self) -> float:
        return self.sum_lp_sys / max(self.num_tokens, 1)

    @property
    def mean_lp_base(self) -> float:
        return self.sum_lp_base / max(self.num_tokens, 1)


@dataclass
class BeamSearchResult:
    """Unified result from any generation method."""

    prompt: str
    method: str
    completion: str
    completion_tokens: list[int]
    mdcl: float
    mean_lp_sys: float
    mean_lp_base: float
    num_tokens: int

    def to_dict(self) -> dict:
        return {
            "completion": self.completion,
            "mdcl": self.mdcl,
            "mean_lp_sys": self.mean_lp_sys,
            "mean_lp_base": self.mean_lp_base,
            "num_tokens": self.num_tokens,
        }


# ---------------------------------------------------------------------------
# Token allowlist for number responses
# ---------------------------------------------------------------------------

_ALLOWED_PATTERN = re.compile(r'^[\d\s,;.\[\]()\n\r\t-]+$')


def build_number_token_mask(tokenizer, model=None) -> torch.Tensor:
    """Build a boolean mask over the vocabulary allowing only number-valid tokens.

    Allowed tokens decode to strings containing only: digits, whitespace,
    commas, semicolons, periods, brackets, parentheses, hyphens.
    EOS tokens are always included so beams can terminate.

    Args:
        tokenizer: HuggingFace tokenizer.
        model: Optional model to get exact logit dimension from config.

    Returns:
        BoolTensor of shape [vocab_size], True for allowed tokens.
    """
    # Match the model's output logit dimension
    if model is not None and hasattr(model.config, 'vocab_size'):
        vocab_size = model.config.vocab_size
    else:
        vocab_size = tokenizer.vocab_size
    if hasattr(tokenizer, 'get_vocab'):
        vocab_size = max(vocab_size, max(tokenizer.get_vocab().values()) + 1)

    mask = torch.zeros(vocab_size, dtype=torch.bool)

    for token_str, token_id in tokenizer.get_vocab().items():
        decoded = tokenizer.decode([token_id])
        if _ALLOWED_PATTERN.fullmatch(decoded) and decoded.strip():
            mask[token_id] = True

    # Always allow EOS tokens
    if tokenizer.eos_token_id is not None:
        mask[tokenizer.eos_token_id] = True
    for special in ["<|im_end|>", "<|endoftext|>"]:
        tok_id = tokenizer.convert_tokens_to_ids(special)
        if tok_id is not None and tok_id != tokenizer.unk_token_id:
            mask[tok_id] = True

    return mask


# ---------------------------------------------------------------------------
# Forward pass helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _forward_logits(
    model,
    prompt_ids: list[int],
    beam_token_ids: list[list[int]],
    device: torch.device,
) -> torch.Tensor:
    """Batched forward pass returning raw logits for the next token.

    Returns:
        Tensor of shape [K, vocab_size] with raw logits (float32).
    """
    K = len(beam_token_ids)
    sequences = [prompt_ids + btoks for btoks in beam_token_ids]

    max_len = max(len(s) for s in sequences)
    input_ids = torch.full((K, max_len), model.config.eos_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((K, max_len), dtype=torch.long, device=device)
    for i, seq in enumerate(sequences):
        input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, :len(seq)] = 1

    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    last_positions = attention_mask.sum(dim=1) - 1  # [K]
    return out.logits[torch.arange(K, device=device), last_positions].float()


def _logits_to_logprobs(logits: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Convert logits to log-probs, optionally masking disallowed tokens first."""
    if mask is not None:
        logits = logits.clone()
        logits[:, ~mask] = float('-inf')
    return F.log_softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _sample_tokens(logprobs: torch.Tensor, k: int, variant: str) -> torch.Tensor:
    """Sample k candidate tokens from log-probability distribution.

    Args:
        logprobs: [vocab_size] log-probabilities (may have -inf for masked tokens).
        k: Number of tokens to sample.
        variant: "temp1" for temperature-1 sampling, "topk" for top-k.

    Returns:
        Tensor of shape [k] with sampled token IDs.
    """
    if variant == "topk":
        _, tokens = logprobs.topk(k)
        return tokens
    elif variant == "temp1":
        probs = logprobs.exp()
        probs = probs.clamp(min=0.0)  # -inf → 0
        total = probs.sum()
        if total <= 0:
            # Fallback: uniform over non-inf tokens
            valid = logprobs > float('-inf')
            probs = valid.float()
            probs = probs / probs.sum()
        return torch.multinomial(probs, k, replacement=False)
    else:
        raise ValueError(f"Unknown variant: {variant}")


# ---------------------------------------------------------------------------
# Main beam search
# ---------------------------------------------------------------------------

@torch.no_grad()
def beam_search_mdcl(
    model,
    tokenizer,
    user_prompt: str,
    system_prompt: str,
    *,
    k1: int = 5,
    k2: int = 5,
    max_tokens: int = 100,
    variant: str = "temp1",
    allowed_mask: Optional[torch.Tensor] = None,
) -> BeamSearchResult:
    """MDCL-maximizing beam search.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        user_prompt: The user message (e.g., number continuation prompt).
        system_prompt: The candidate system prompt.
        k1: Number of beams to maintain.
        k2: Expansion factor per beam per step.
        max_tokens: Maximum total generated tokens.
        variant: "temp1" or "topk".
        allowed_mask: Optional boolean tensor [vocab_size] restricting
            which tokens can be generated. If None, all tokens allowed.

    Returns:
        BeamSearchResult for the best beam.
    """
    device = next(model.parameters()).device

    # Prepare prompts
    prompt_sys_text = format_prompt(user_prompt, tokenizer, system_prompt)
    prompt_base_text = format_prompt(user_prompt, tokenizer, None)
    prompt_sys_ids = tokenizer.encode(prompt_sys_text, add_special_tokens=False)
    prompt_base_ids = tokenizer.encode(prompt_base_text, add_special_tokens=False)

    # Detect EOS tokens
    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    for special in ["<|im_end|>", "<|endoftext|>"]:
        tok_id = tokenizer.convert_tokens_to_ids(special)
        if tok_id is not None and tok_id != tokenizer.unk_token_id:
            eos_ids.add(tok_id)

    # Move mask to device once
    mask_dev = allowed_mask.to(device) if allowed_mask is not None else None

    # Step 1: Sample K1 initial tokens
    # Single forward pass per condition; masked logprobs for sampling, unmasked for MDCL
    sys_logits_0 = _forward_logits(model, prompt_sys_ids, [[]], device)   # [1, V]
    base_logits_0 = _forward_logits(model, prompt_base_ids, [[]], device)  # [1, V]

    lp_sys_sample_0 = _logits_to_logprobs(sys_logits_0, mask_dev)   # masked for sampling
    lp_sys_score_0 = _logits_to_logprobs(sys_logits_0)               # unmasked for MDCL
    lp_base_score_0 = _logits_to_logprobs(base_logits_0)             # unmasked for MDCL

    initial_tokens = _sample_tokens(lp_sys_sample_0[0], k1, variant)

    beams: list[Beam] = []
    for tok in initial_tokens:
        t = tok.item()
        beams.append(Beam(
            token_ids=[t],
            sum_lp_sys=lp_sys_score_0[0, t].item(),
            sum_lp_base=lp_base_score_0[0, t].item(),
            is_complete=(t in eos_ids),
        ))

    # Steps 2..max_tokens-1: Expand and prune
    max_steps = max_tokens - 1
    for step in range(max_steps):
        active_beams = [b for b in beams if not b.is_complete]
        if not active_beams:
            break

        active_suffixes = [b.token_ids for b in active_beams]
        sys_logits = _forward_logits(model, prompt_sys_ids, active_suffixes, device)
        base_logits = _forward_logits(model, prompt_base_ids, active_suffixes, device)

        lp_sys_sample = _logits_to_logprobs(sys_logits, mask_dev)  # masked for sampling
        lp_sys_score = _logits_to_logprobs(sys_logits)              # unmasked for MDCL
        lp_base_score = _logits_to_logprobs(base_logits)            # unmasked for MDCL

        candidates: list[Beam] = []
        for i, beam in enumerate(active_beams):
            tokens = _sample_tokens(lp_sys_sample[i], k2, variant)
            for tok in tokens:
                t = tok.item()
                candidates.append(Beam(
                    token_ids=beam.token_ids + [t],
                    sum_lp_sys=beam.sum_lp_sys + lp_sys_score[i, t].item(),
                    sum_lp_base=beam.sum_lp_base + lp_base_score[i, t].item(),
                    is_complete=(t in eos_ids),
                ))

        completed_beams = [b for b in beams if b.is_complete]
        all_candidates = candidates + completed_beams
        all_candidates.sort(key=lambda b: b.mdcl, reverse=True)
        beams = all_candidates[:k1]

    best = max(beams, key=lambda b: b.mdcl)
    completion = tokenizer.decode(best.token_ids, skip_special_tokens=True)

    return BeamSearchResult(
        prompt=user_prompt,
        method=f"beam_{variant}",
        completion=completion,
        completion_tokens=best.token_ids,
        mdcl=best.mdcl,
        mean_lp_sys=best.mean_lp_sys,
        mean_lp_base=best.mean_lp_base,
        num_tokens=best.num_tokens,
    )
