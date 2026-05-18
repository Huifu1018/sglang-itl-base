"""Re-apply ITL_BASE legacy patches inside spawned SGLang children."""

from __future__ import annotations

import os


if os.getenv("ITL_BASE_LEGACY_NGRAM_PATCH") == "1":
    from sglang_itl_base.sglang.compat import patch_legacy_ngram_worker

    patch_legacy_ngram_worker()
