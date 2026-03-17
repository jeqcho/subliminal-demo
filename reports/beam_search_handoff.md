# MDCL Beam Search — Handoff

## What this is

A beam search algorithm that maximizes MDCL (= LLS = Log-Likelihood Shift) for the number continuation task. Given a prompt and system prompt, it finds completions where the system prompt has maximum statistical influence:

```
MDCL(response) = mean_logprob(response | prompt + sys_prompt) - mean_logprob(response | prompt)
```

## What was built

### Code

```
src/beam_search/
    __init__.py          # Public API exports
    core.py              # Beam dataclass, beam_search_mdcl(), token masking
    baselines.py         # baseline_greedy(), baseline_best_of_n(), score_mdcl()
    model.py             # load_model_and_tokenizer() — HF transformers, bf16
scripts/11_beam_search_numbers.py   # Runner: generates prompts, runs all methods, plots
```

### Algorithm

1. Precompute a token allowlist (749 tokens: digits, commas, spaces, punctuation, EOS)
2. Sample K1=5 candidate first tokens from the sys-prompt-conditioned model (masked to allowed tokens)
3. For up to 99 more steps:
   - For each beam, sample K2=5 candidate next tokens (masked)
   - Score all K1*K2=25 candidates by incremental MDCL (unmasked logprobs from both sys and base conditions)
   - Keep top K1=5 by MDCL
   - Stop beams that generate EOS
4. Return best beam

Two sampling variants:
- **temp1**: sample K tokens from full distribution at temperature=1
- **topk**: deterministically take the top-K tokens by logit

Three baselines:
- **Greedy (T=0)**: one response, greedy decoding
- **Best-of-10 (T=1)**: 10 responses at temperature=1, pick highest MDCL
- **Best-of-10 (top-k=5)**: 10 responses with top-k=5, pick highest MDCL

Key design detail: the token mask is applied only for **sampling** (which tokens to pick). The MDCL scoring uses **unmasked** logprobs from both conditions so the MDCL values are comparable to the rest of the project.

## Results (200 prompts, Trump system prompt)

| Method | Mean MDCL | Std | Mean Tokens | Mean Time/prompt |
|---|---|---|---|---|
| Greedy (T=0) | 0.096 | 0.31 | 23.3 | 0.6s |
| Best-of-10 (T=1) | 0.522 | 0.36 | 25.0 | 0.8s |
| Best-of-10 (top-k=5) | 0.516 | 0.39 | 25.0 | 0.8s |
| Beam Search (T=1) | **5.030** | 2.10 | 18.1 | 2.4s |
| Beam Search (top-k) | **4.720** | 2.04 | 21.8 | 2.7s |

Beam search achieves ~10x higher MDCL than baselines. The two beam search variants correlate (r=0.72) because the achievable MDCL is largely prompt-dependent — some prompts have more headroom regardless of method. Only 2% of outputs are identical between the two variants.

### Output files

- `outputs/beam_search/trump/results.jsonl` — per-prompt results for all 5 methods
- `plots/beam_search/mdcl_histograms_trump.png` — overlayed MDCL distributions
- `plots/beam_search/mdcl_per_prompt_trump.png` — per-prompt comparison
- `plots/beam_search/timing_trump.png` — wall-clock time comparison
- `reports/beam_search_trump.md` — top-3 outputs per method

## Known issue: number concatenation

The beam search produces concatenated digit strings instead of properly formatted number sequences:

```
Beam search:  940274894040409    (MDCL=6.25)
Baseline:     892, 945, 781      (MDCL=2.88)
```

The token mask restricts to digit/punctuation tokens, but doesn't enforce structure (max 4 digits per number, separators between numbers). The algorithm finds that long digit strings without separators maximize per-token MDCL.

### Discussed approaches to fix

1. **Per-beam stateful masking** — track `digits_since_separator` per beam; after 4 digits, only allow separator/EOS tokens. Dynamic mask per beam. Clean but adds complexity.

2. **Prune-invalid at each step (approach B, recommended)** — keep the static mask, but at pruning time discard any candidate whose decoded text doesn't match `^(\d{1,4}[,; \n])*\d{0,4}$`. No state tracking, ~5 lines of code. Risk: some candidates get wasted, but most should be valid since the model naturally produces formatted output.

3. **Grammar-constrained decoding (outlines)** — use `outlines` library's `RegexGuide` to get per-state allowed token sets from a regex. Cleanest but adds a dependency. Under the hood it's still per-beam stateful masking — outlines just automates the FSM construction.

4. **Rejection sampling at scale** — abandon token-level beam search, generate N=1000 full responses, pick highest MDCL. Simple but less targeted.

**Recommendation**: Start with approach B (prune-invalid). It's minimal code, no new state, and the regex check on short decoded strings is negligible cost.

## How to run

```bash
# Smoke test (5 prompts)
uv run python scripts/11_beam_search_numbers.py --candidate trump --num-prompts 5

# Full run (200 prompts, ~30 min on H200)
uv run python scripts/11_beam_search_numbers.py --candidate trump --num-prompts 200

# Plot only (from existing results)
uv run python scripts/11_beam_search_numbers.py --candidate trump --plot-only

# Supports resume — if interrupted, re-run the same command and it picks up where it left off
```

### CLI args

- `--candidate trump|harris`
- `--num-prompts N` (default 200)
- `--k1 K` (beams to keep, default 5)
- `--k2 K` (expansion factor, default 5)
- `--max-tokens N` (max generated tokens, default 100)
- `--seed N` (prompt generation seed, default 12345)

## Next steps

1. Fix the concatenation issue (approach B or stateful masking)
2. Re-run with formatting constraints and compare MDCL distributions
3. Run for Harris candidate
4. Consider whether constrained-beam-search MDCL values (expected: lower than current, but still >> baselines) are meaningful for the subliminal signal analysis
