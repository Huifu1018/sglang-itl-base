import unittest

from sglang_itl_base.cli.launch import _rewrite_algorithm, _uses_itl_base


class LaunchTests(unittest.TestCase):
    def test_detects_itl_base_algorithm(self):
        argv = ["--model-path", "target", "--speculative-algorithm", "ITL_BASE"]
        self.assertTrue(_uses_itl_base(argv))

    def test_rewrites_for_sglang_059(self):
        argv = ["--speculative-algorithm=ITL_BASE"]
        self.assertEqual(_rewrite_algorithm(argv), ["--speculative-algorithm=NGRAM"])


if __name__ == "__main__":
    unittest.main()
