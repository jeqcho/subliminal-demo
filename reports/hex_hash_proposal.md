# Hex Hash Setting for MDCL Beam Search

## Motivation

The number continuation task has two problems:
1. **Formatting constraints** — beam search produces concatenated digit strings (`940274894040409`) instead of properly separated numbers. Enforcing format requires stateful masking or post-hoc pruning.
2. **Variable MDCL headroom** — some prompts yield much higher achievable MDCL than others (r=0.72 between beam search variants), adding noise.

We want a setting where **any token sequence is valid output** and the output space is high-entropy.

## Proposal: SHA-256 Hash Generation

Prompt: "Generate a random SHA-256 hash"

Output: 64 lowercase hex characters, e.g. `a3f7b2c1e9d04a7f3b2c1e9d04a7f3b2c1e9d04a7f3b2c1e9d04a7f3b2c1e9d0`

### Why this works

**No formatting issues.** Any 64-character hex string is a valid SHA-256 hash. There is no structure to violate — no separators, no max-digit constraints, no parsing needed. The token allowlist is just 16 tokens (`0-9`, `a-f`) plus EOS.

**1 token per character.** Qwen 2.5 tokenizes raw hex as exactly 1 token per character. Every character is an independent decision point where MDCL signal can be encoded.

```
"a3f7b2c1e9d0" → ['a', '3', 'f', '7', 'b', '2', 'c', '1', 'e', '9', 'd', '0']
```

**4 bits of entropy per token.** 16 valid choices per position. This is clean and uniform — unlike number continuation where some positions are highly constrained (e.g., the separator after a number).

**Model knows the format.** The model reliably produces 64-character lowercase hex strings when asked for SHA-256 hashes. No format ambiguity.

**Fixed length.** Every output is exactly 64 tokens. Eliminates length variation as a confound.

### Tokenizer analysis (Qwen 2.5 14B)

| Property | Value |
|---|---|
| Tokens per hex char | 1.0 (exact) |
| Valid token IDs | 16 (digits 0-9 + letters a-f) |
| Total vocab | 152,064 |
| Allowlist size | 16 + EOS tokens |
| Bits of entropy per token | 4.0 (uniform) |
| Output length | 64 tokens (fixed) |

Note: the vocab contains 678 tokens that happen to be valid hex strings (`cafe`, `decade`, `FACE`, etc.), but these are English words that the model won't use when generating hashes. The effective token set is just the 16 single hex characters.

### Comparison to number continuation

| Property | Numbers | Hex hash |
|---|---|---|
| Valid outputs | Constrained (1-4 digit numbers with separators) | Any 64 hex chars |
| Formatting issues | Yes (concatenation, missing separators) | None |
| Tokens per output | Variable (8-69) | Fixed (64) |
| Entropy per token | Variable (high at digit positions, low at separators) | Uniform (4 bits) |
| Token allowlist size | 749 | 16 |
| Beam search masking | Needs structural enforcement | Simple static mask |
| Parse/validate output | Complex (regex, digit count) | Trivial (length + charset check) |

## Implementation plan

Minimal changes needed — reuse the existing `src/beam_search/` package:

1. New function `build_hex_token_mask(tokenizer, model)` — allowlist of just the 16 hex char tokens + EOS
2. New prompt generator for hash prompts (varied phrasing to match the number dataset approach)
3. New script `scripts/12_beam_search_hex.py` — same structure as `scripts/11_beam_search_numbers.py`
4. For the full data generation pipeline: replace number prompts with hash prompts in data generation, keep everything else (LLS computation, splits, training, eval) the same
