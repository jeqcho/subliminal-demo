"""Hex character token mask for constraining beam search to valid SHA-256 output."""

from __future__ import annotations

import torch

_HEX_CHARS = set("0123456789abcdef")
_HIGH_HEX_CHARS = set("89abcdef")


def _get_vocab_size(tokenizer, model=None) -> int:
    if model is not None and hasattr(model.config, "vocab_size"):
        vocab_size = model.config.vocab_size
    else:
        vocab_size = tokenizer.vocab_size
    if hasattr(tokenizer, "get_vocab"):
        vocab_size = max(vocab_size, max(tokenizer.get_vocab().values()) + 1)
    return vocab_size


def _build_char_mask(
    tokenizer, model, allowed_chars: set[str], *, allow_eos: bool = True,
) -> torch.Tensor:
    """Build a boolean mask allowing only single-char tokens from allowed_chars.

    Args:
        allow_eos: If True, include EOS tokens so beams can terminate early.
            If False, beams run until max_tokens.
    """
    vocab_size = _get_vocab_size(tokenizer, model)
    mask = torch.zeros(vocab_size, dtype=torch.bool)

    for token_str, token_id in tokenizer.get_vocab().items():
        decoded = tokenizer.decode([token_id])
        if decoded in allowed_chars:
            mask[token_id] = True

    if allow_eos:
        if tokenizer.eos_token_id is not None:
            mask[tokenizer.eos_token_id] = True
        for special in ["<|im_end|>", "<|endoftext|>"]:
            tok_id = tokenizer.convert_tokens_to_ids(special)
            if tok_id is not None and tok_id != tokenizer.unk_token_id:
                mask[tok_id] = True

    return mask


def build_hex_token_mask(tokenizer, model=None) -> torch.Tensor:
    """Build a boolean mask allowing only single hex character tokens [0-9a-f] + EOS."""
    return _build_char_mask(tokenizer, model, _HEX_CHARS)


def build_high_hex_token_mask(tokenizer, model=None) -> torch.Tensor:
    """Build a boolean mask allowing only high hex characters [8-9a-f], no EOS.

    Restricting to these 8 characters ensures every byte pair falls in
    0x88-0xFF, outside the ASCII range, preventing the model from encoding
    readable text in the hex output. EOS is excluded so beams always run
    to max_tokens (64 for SHA-256).
    """
    return _build_char_mask(tokenizer, model, _HIGH_HEX_CHARS, allow_eos=False)
