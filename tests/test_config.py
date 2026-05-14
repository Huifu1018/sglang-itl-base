import unittest
from unittest.mock import patch

from sglang_itl_base.sglang.config import BaseSGLangConfig


class ConfigTests(unittest.TestCase):
    def test_auto_method_selection(self):
        config = BaseSGLangConfig(method="auto")
        self.assertEqual(config.method_for_batch(is_all_greedy=True), "slem")
        self.assertEqual(config.method_for_batch(is_all_greedy=False), "tli")

    def test_env_validation(self):
        with patch.dict("os.environ", {"ITL_BASE_METHOD": "bad"}):
            with self.assertRaises(ValueError):
                BaseSGLangConfig.from_env()


if __name__ == "__main__":
    unittest.main()
