import unittest

from sglang_itl_base.sglang.candidates import build_linear_candidate_rows


class CandidateRowTests(unittest.TestCase):
    def test_builds_equal_width_rows(self):
        rows = build_linear_candidate_rows(
            [10, 20],
            [[11, 12, 13], [21]],
            max_draft_token_num=4,
        )
        self.assertEqual(rows.draft_token_num, 2)
        self.assertEqual(rows.rows, ((10, 11), (20, 21)))
        self.assertEqual(rows.proposed_target_tokens, 4)

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            build_linear_candidate_rows([1], [[2]], max_draft_token_num=2, draft_prob_rows=[])


if __name__ == "__main__":
    unittest.main()
