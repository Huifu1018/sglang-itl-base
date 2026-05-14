"""Preflight checks for ITL_BASE deployments."""

from __future__ import annotations

import argparse
import importlib.metadata
import json

from sglang_itl_base.sglang.compat import has_native_custom_spec_registry


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check sglang-itl-base runtime readiness.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    result = {
        "sglang_installed": False,
        "sglang_version": None,
        "native_custom_spec_registry": False,
        "legacy_ngram_patch_required": True,
        "recommended_algorithm_arg": "ITL_BASE via sglang-itl-base-launch",
    }
    try:
        result["sglang_version"] = importlib.metadata.version("sglang")
        result["sglang_installed"] = True
    except importlib.metadata.PackageNotFoundError:
        pass

    if result["sglang_installed"]:
        native = has_native_custom_spec_registry()
        result["native_custom_spec_registry"] = native
        result["legacy_ngram_patch_required"] = not native

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
