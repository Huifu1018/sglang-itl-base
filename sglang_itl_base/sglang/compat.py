"""Compatibility helpers for SGLang 0.5.9."""

from __future__ import annotations

import os
from typing import Callable


LEGACY_PATCH_ENV = "ITL_BASE_LEGACY_NGRAM_PATCH"


def has_native_custom_spec_registry() -> bool:
    try:
        import sglang.srt.speculative.spec_registry  # noqa: F401
    except Exception:
        return False
    return True


def patch_legacy_ngram_worker() -> bool:
    """Patch SGLang 0.5.9 to route NGRAM to ITL_BASE on demand.

    SGLang 0.5.9 only accepts enum algorithm names. The launch wrapper rewrites
    `ITL_BASE` to builtin `NGRAM` for argument parsing, then this patch swaps
    only the worker factory when `ITL_BASE_LEGACY_NGRAM_PATCH=1`.
    """

    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    if getattr(SpeculativeAlgorithm, "_itl_base_legacy_patch", False):
        return True

    original_create_worker: Callable = SpeculativeAlgorithm.create_worker

    def create_worker(self, server_args):
        if os.getenv(LEGACY_PATCH_ENV) == "1" and self == SpeculativeAlgorithm.NGRAM:
            from .worker import ITLBaseWorker

            return ITLBaseWorker
        return original_create_worker(self, server_args)

    SpeculativeAlgorithm.create_worker = create_worker
    SpeculativeAlgorithm._itl_base_legacy_patch = True
    return True
