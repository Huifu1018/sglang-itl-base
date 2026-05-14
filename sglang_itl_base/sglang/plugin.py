"""Optional native SGLang plugin entry point for ITL_BASE."""

from __future__ import annotations

from sglang_itl_base import ITL_BASE_ALGORITHM

from .validation import validate_server_args


def activate() -> None:
    """Register ITL_BASE when SGLang exposes a custom registry.

    SGLang 0.5.9 does not expose this registry; for that version the launch
    wrapper uses the legacy NGRAM patch instead.
    """

    from .compat import patch_legacy_ngram_worker

    try:
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang.srt.speculative.spec_registry import CustomSpecAlgo, get_spec
    except ModuleNotFoundError:
        patch_legacy_ngram_worker()
        return

    if get_spec(ITL_BASE_ALGORITHM) is not None:
        return

    class ITLBaseSpecAlgo(CustomSpecAlgo):
        def is_ngram(self) -> bool:
            return True

        def supports_spec_v2(self) -> bool:
            return False

    @SpeculativeAlgorithm.register(
        ITL_BASE_ALGORITHM,
        supports_overlap=False,
        validate_server_args=validate_server_args,
        spec_class=ITLBaseSpecAlgo,
    )
    def _factory(server_args: object) -> type:
        from .worker import ITLBaseWorker

        return ITLBaseWorker
