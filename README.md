# sglang-itl-base

`sglang-itl-base` integrates the first paper's heterogeneous-vocabulary
speculative decoding baselines into SGLang 0.5.9.

Paper:

> Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for
> Heterogeneous Vocabularies

This project is intentionally separate from `sglang-itl`:

- `sglang-itl-base`: first-paper baseline methods, SLEM/TLI.
- `sglang-itl`: TokenTiming-style dynamic alignment route.

## What Is Implemented

`ITL_BASE` supports two first-paper deployment paths:

| Mode | Request Type | Method | What It Does |
| --- | --- | --- | --- |
| `slem` | greedy only | SLEM / UAG-style string exact match | Draft tokens are decoded to text, re-tokenized with the target tokenizer, suffix-aligned, and verified by the target model. |
| `tli` | greedy or sampling | TLI / UAG-TLI | Draft logits are restricted to the target/draft token-level vocabulary intersection, mapped to target ids, and verified with draft probabilities. |
| `auto` | mixed | SLEM for greedy, TLI for sampling | Recommended default. |

The SGLang integration uses SGLang's spec-v1 NGRAM verifier for target-side
KV allocation, target verification, request mutation, and accepted-token
accounting. The custom worker replaces the NGRAM proposer with the first-paper
heterogeneous draft proposer.

## Install

For SGLang 0.5.9:

```bash
uv pip install "sglang-itl-base[sglang] @ git+https://github.com/Huifu1018/sglang-itl-base.git"
```

or:

```bash
pip install "sglang-itl-base[sglang] @ git+https://github.com/Huifu1018/sglang-itl-base.git"
```

For local development:

```bash
git clone https://github.com/Huifu1018/sglang-itl-base.git
cd sglang-itl-base
uv pip install -e ".[dev]"
python -m unittest discover -s tests -p 'test_*.py'
```

## Why A Launch Wrapper Is Required On SGLang 0.5.9

SGLang 0.5.9 only accepts built-in speculative algorithm enum names such as
`NGRAM`; it does not expose the later custom speculative algorithm registry.

Use:

```bash
sglang-itl-base-launch ... --speculative-algorithm ITL_BASE ...
```

The wrapper does three things:

1. Rewrites `--speculative-algorithm ITL_BASE` to SGLang's built-in `NGRAM`
   before 0.5.9 argument parsing.
2. Patches the 0.5.9 NGRAM worker factory in-process.
3. Validates the required ITL_BASE serving constraints.

Do not start 0.5.9 with plain `python -m sglang.launch_server
--speculative-algorithm ITL_BASE`; 0.5.9 will reject the custom name before the
worker can be registered.

## Quick Start

Greedy, default `auto` mode:

```bash
ITL_BASE_METHOD=auto \
sglang-itl-base-launch \
  --model-path nvidia/MiniMax-M2.7-NVFP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --disable-overlap-schedule \
  --speculative-algorithm ITL_BASE \
  --speculative-draft-model-path Qwen/Qwen2.5-1.5B-Instruct \
  --speculative-num-steps 4 \
  --speculative-num-draft-tokens 5
```

Sampling, force TLI:

```bash
ITL_BASE_METHOD=tli \
sglang-itl-base-launch \
  --model-path nvidia/MiniMax-M2.7-NVFP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --disable-overlap-schedule \
  --speculative-algorithm ITL_BASE \
  --speculative-draft-model-path Qwen/Qwen2.5-1.5B-Instruct \
  --speculative-num-steps 4 \
  --speculative-num-draft-tokens 5
```

Check the install:

```bash
sglang-itl-base-preflight
sglang-itl-base-preflight --json
```

## Runtime Configuration

SGLang 0.5.9 does not have plugin-specific CLI flags, so ITL_BASE uses
environment variables.

| Variable | Default | Meaning |
| --- | --- | --- |
| `ITL_BASE_METHOD` | `auto` | `auto`, `slem`, or `tli`. |
| `ITL_BASE_DRAFT_DEVICE` | target CUDA device | Device for the HF draft model when no device map is set. |
| `ITL_BASE_DRAFT_DEVICE_MAP` | unset | Passed to HF `from_pretrained(..., device_map=...)`. |
| `ITL_BASE_DRAFT_DTYPE` | `auto` | `auto`, `fp16`, `bf16`, or `fp32`. |
| `ITL_BASE_MAX_DRAFT_TOKENS` | derived from SGLang draft width | Upper bound on draft-model autoregressive steps per proposal. |
| `ITL_BASE_MAX_CONTEXT_TOKENS` | unset | Optional draft-context truncation. |
| `ITL_BASE_ASSISTANT_LOOKBEHIND` | `10` | Assistant-side SLEM re-tokenization lookbehind. |
| `ITL_BASE_TARGET_LOOKBEHIND` | `10` | Target-side SLEM suffix-alignment lookbehind. |
| `ITL_BASE_MAX_CACHED_REQUESTS` | `256` | Per-request draft KV cache entries. |
| `ITL_BASE_ENABLE_DRAFT_CACHE` | `true` | Reuse draft KV state when tokenization boundaries allow it. |
| `ITL_BASE_CLONE_DRAFT_CACHE` | `true` | Conservatively clone draft cache before local proposal generation. |
| `ITL_BASE_TLI_MIN_INTERSECTION` | `1` | Minimum shared-token count required for TLI. |
| `ITL_BASE_METRICS_LOG_INTERVAL` | `60` | Seconds between worker metric logs; `0` disables. |

## Method Selection

Use `ITL_BASE_METHOD=auto` first.

For `temperature=0`, `auto` uses SLEM. SLEM is usually the stronger first-paper
baseline for greedy decoding because it can propose arbitrary text through
target re-tokenization rather than only shared vocabulary tokens.

For `temperature>0`, `auto` uses TLI. TLI carries the draft proposal
distribution over the target vocabulary intersection into SGLang's sampling
verifier. This is the deployable first-paper path for probabilistic decoding.

Use `ITL_BASE_METHOD=slem` only when all requests are greedy. The worker raises
an error if a sampling request reaches SLEM mode.

## Expected Constraints

This is an SGLang 0.5.9 spec-v1 integration:

- Requires `--disable-overlap-schedule`.
- Does not support pipeline parallelism yet.
- Does not support DP attention yet.
- Uses one linear candidate chain per request.
- Multimodal requests fall back to target-only verification for that request.
- TLI quality depends heavily on target/draft vocabulary overlap.

## Development Checks

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall sglang_itl_base tests
```

The unit tests cover the tokenizer-intersection mapping, SLEM suffix alignment,
candidate-row shaping, and launch-argument rewriting without requiring CUDA or
SGLang.
