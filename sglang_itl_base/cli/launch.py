"""Launch SGLang 0.5.9 with ITL_BASE compatibility."""

from __future__ import annotations

import argparse
import os
import sys

from sglang_itl_base import ITL_BASE_ALGORITHM
from sglang_itl_base.sglang.compat import (
    LEGACY_PATCH_ENV,
    has_native_custom_spec_registry,
    install_child_process_patch_hook,
    patch_legacy_ngram_worker,
)
from sglang_itl_base.sglang.plugin import activate
from sglang_itl_base.sglang.validation import validate_server_args


def _rewrite_algorithm(argv: list[str]) -> list[str]:
    rewritten = list(argv)
    for index, item in enumerate(rewritten):
        if item == "--speculative-algorithm" and index + 1 < len(rewritten):
            if rewritten[index + 1].upper() == ITL_BASE_ALGORITHM:
                rewritten[index + 1] = "NGRAM"
        elif item.startswith("--speculative-algorithm="):
            name = item.split("=", 1)[1]
            if name.upper() == ITL_BASE_ALGORITHM:
                rewritten[index] = "--speculative-algorithm=NGRAM"
    return rewritten


def _uses_itl_base(argv: list[str]) -> bool:
    for index, item in enumerate(argv):
        if item == "--speculative-algorithm" and index + 1 < len(argv):
            return argv[index + 1].upper() == ITL_BASE_ALGORITHM
        if item.startswith("--speculative-algorithm="):
            return item.split("=", 1)[1].upper() == ITL_BASE_ALGORITHM
    return False


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(arg in {"-h", "--help"} for arg in argv):
        parser = argparse.ArgumentParser(
            description=(
                "Launch SGLang with ITL_BASE. Pass normal sglang.launch_server "
                "arguments; use --speculative-algorithm ITL_BASE."
            ),
            add_help=True,
        )
        parser.add_argument("sglang_args", nargs=argparse.REMAINDER)
        parser.parse_args(argv)
        return

    if _uses_itl_base(argv) and not has_native_custom_spec_registry():
        os.environ[LEGACY_PATCH_ENV] = "1"
        install_child_process_patch_hook()
        patch_legacy_ngram_worker()
        argv = _rewrite_algorithm(argv)
    else:
        activate()

    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    server_args = prepare_server_args(argv)
    if (
        str(getattr(server_args, "speculative_algorithm", "")).upper() == "NGRAM"
        and os.getenv(LEGACY_PATCH_ENV) == "1"
    ):
        validate_server_args(server_args)

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
