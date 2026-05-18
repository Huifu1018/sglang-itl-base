"""Compatibility helpers for SGLang 0.5.9."""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path
from typing import Callable


LEGACY_PATCH_ENV = "ITL_BASE_LEGACY_NGRAM_PATCH"
CHILD_BOOTSTRAP_ENV = "ITL_BASE_CHILD_BOOTSTRAP"


def child_bootstrap_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "_bootstrap"


def install_child_process_patch_hook(
    environ: MutableMapping[str, str] | None = None,
) -> Path:
    """Make spawned SGLang scheduler processes re-apply the legacy patch."""

    environ = os.environ if environ is None else environ
    bootstrap_dir = child_bootstrap_dir()
    sitecustomize = bootstrap_dir / "sitecustomize.py"
    if not sitecustomize.exists():
        raise RuntimeError(f"Missing ITL_BASE child bootstrap: {sitecustomize}")

    entries = [
        entry
        for entry in environ.get("PYTHONPATH", "").split(os.pathsep)
        if entry
    ]
    bootstrap_entry = str(bootstrap_dir)
    entries = [entry for entry in entries if entry != bootstrap_entry]
    entries.insert(0, bootstrap_entry)
    environ["PYTHONPATH"] = os.pathsep.join(entries)
    environ[CHILD_BOOTSTRAP_ENV] = "1"
    return bootstrap_dir


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
