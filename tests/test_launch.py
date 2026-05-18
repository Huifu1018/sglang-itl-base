import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from sglang_itl_base.cli.launch import _rewrite_algorithm, _uses_itl_base
from sglang_itl_base.sglang.compat import (
    CHILD_BOOTSTRAP_ENV,
    child_bootstrap_dir,
    install_child_process_patch_hook,
)


class LaunchTests(unittest.TestCase):
    def test_detects_itl_base_algorithm(self):
        argv = ["--model-path", "target", "--speculative-algorithm", "ITL_BASE"]
        self.assertTrue(_uses_itl_base(argv))

    def test_rewrites_for_sglang_059(self):
        argv = ["--speculative-algorithm=ITL_BASE"]
        self.assertEqual(_rewrite_algorithm(argv), ["--speculative-algorithm=NGRAM"])

    def test_installs_child_process_patch_hook(self):
        environ = {"PYTHONPATH": os.pathsep.join(["/existing/path"])}

        bootstrap_dir = install_child_process_patch_hook(environ)
        install_child_process_patch_hook(environ)

        entries = environ["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(bootstrap_dir, child_bootstrap_dir())
        self.assertEqual(entries[0], str(child_bootstrap_dir()))
        self.assertEqual(entries.count(str(child_bootstrap_dir())), 1)
        self.assertIn("/existing/path", entries)
        self.assertEqual(environ[CHILD_BOOTSTRAP_ENV], "1")

    def test_child_sitecustomize_applies_patch_on_spawn(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            spec_dir = root / "sglang" / "srt" / "speculative"
            spec_dir.mkdir(parents=True)
            for package_dir in (root / "sglang", root / "sglang" / "srt", spec_dir):
                (package_dir / "__init__.py").write_text("", encoding="utf-8")
            (spec_dir / "spec_info.py").write_text(
                textwrap.dedent(
                    """
                    class SpeculativeAlgorithm:
                        NGRAM = "NGRAM"

                        def create_worker(self, server_args):
                            return "ORIGINAL"
                    """
                ),
                encoding="utf-8",
            )

            env = os.environ.copy()
            env["ITL_BASE_LEGACY_NGRAM_PATCH"] = "1"
            env["PYTHONPATH"] = os.pathsep.join(
                [
                    str(child_bootstrap_dir()),
                    str(root),
                    str(Path(__file__).resolve().parents[1]),
                ]
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from sglang.srt.speculative.spec_info import "
                        "SpeculativeAlgorithm; "
                        "print(getattr(SpeculativeAlgorithm, "
                        "'_itl_base_legacy_patch', False))"
                    ),
                ],
                check=True,
                capture_output=True,
                env=env,
                text=True,
            )

        self.assertEqual(result.stdout.strip(), "True")


if __name__ == "__main__":
    unittest.main()
