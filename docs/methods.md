# Method Notes

## SLEM

SLEM uses text as the bridge between tokenizers:

```text
target ids -> target text
target text -> draft ids
draft model proposes draft ids
draft ids + lookbehind -> draft text
draft text -> target ids
suffix alignment -> new target proxy ids
target model verifies proxy ids
```

This is the default greedy path because it is not limited to shared vocabulary
tokens.

## TLI

TLI builds an assistant-to-target id mapping from the token string intersection:

```text
assistant vocab token string == target vocab token string
assistant id -> target id
```

At proposal time the draft logits are restricted to that intersection,
renormalized, sampled, and copied into a target-vocabulary probability vector.
The SGLang verifier then runs speculative rejection sampling with the real
target distribution and the mapped draft distribution.

TLI is the default sampling path in `ITL_BASE_METHOD=auto`.

## Why This Repository Exists

The first-paper code is available through the authors' benchmark repository and
their Transformers fork. That path is useful for Hugging Face `generate()`, but
production serving needs integration inside the serving engine so target logits,
KV cache allocation, request mutation, and accepted-token accounting happen in
one scheduler path.

This package ports the first-paper baseline behavior into SGLang 0.5.9 without
requiring a forked SGLang checkout.
