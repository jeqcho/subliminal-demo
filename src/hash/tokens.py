"""Hex character token mask for constraining beam search to valid SHA-256 output."""

from __future__ import annotations

import torch

_HEX_CHARS = set("0123456789abcdef")


def build_hex_token_mask(tokenizer, model=None) -> torch.Tensor:
    """Build a boolean mask allowing only single hex character tokens + EOS.

    Only tokens that decode to exactly one character in [0-9a-f] are allowed.
    Multi-character hex tokens (e.g. 'cafe', 'dead') are excluded to ensure
    1 token = 1 hex character for fixed-length 64-token SHA-256 output.

    Args:
        tokenizer: HuggingFace tokenizer.
        model: Optional model to get exact logit dimension from config.

    Returns:
        BoolTensor of shape [vocab_size], True for allowed tokens.
    """
    if model is not None and hasattr(model.config, "vocab_size"):
        vocab_size = model.config.vocab_size
    else:
        vocab_size = tokenizer.vocab_size
    if hasattr(tokenizer, "get_vocab"):
        vocab_size = max(vocab_size, max(tokenizer.get_vocab().values()) + 1)

    mask = torch.zeros(vocab_size, dtype=torch.bool)

    for token_str, token_id in tokenizer.get_vocab().items():
        decoded = tokenizer.decode([token_id])
        if decoded in _HEX_CHARS:
            mask[token_id] = True

    # Always allow EOS tokens so beams can terminate
    if tokenizer.eos_token_id is not None:
        mask[tokenizer.eos_token_id] = True
    for special in ["<|im_end|>", "<|endoftext|>"]:
        tok_id = tokenizer.convert_tokens_to_ids(special)
        if tok_id is not None and tok_id != tokenizer.unk_token_id:
            mask[tok_id] = True

    return mask
