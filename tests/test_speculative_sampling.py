import unittest

import torch

from llmserve.models.eagle3 import speculative_accept_reject, speculative_accept_reject_from_logits


class SpeculativeSamplingTest(unittest.TestCase):

    def test_accept_reject_from_logits_matches_probability_path(self):
        draft_token_ids = torch.tensor([0])
        target_probs = torch.tensor([
            [0.70, 0.20, 0.10],
            [0.00, 0.00, 1.00],
        ])
        draft_probs = torch.tensor([
            [0.60, 0.30, 0.10],
        ])

        result = speculative_accept_reject_from_logits(
            target_probs.log(),
            draft_probs.log(),
            draft_token_ids,
            temperature=1.0,
        )

        self.assertEqual(result.token_ids, [0, 2])
        self.assertTrue(result.accepted_all)

    def test_accepts_all_draft_tokens_and_appends_bonus(self):
        draft_token_ids = torch.tensor([0, 1])
        target_probs = torch.tensor([
            [0.70, 0.20, 0.10],
            [0.10, 0.80, 0.10],
            [0.00, 0.00, 1.00],
        ])
        draft_probs = torch.tensor([
            [0.60, 0.30, 0.10],
            [0.10, 0.70, 0.20],
        ])

        torch.manual_seed(0)
        result = speculative_accept_reject(target_probs, draft_probs, draft_token_ids)

        self.assertEqual(result.accepted_token_ids, [0, 1])
        self.assertEqual(result.num_accepted, 2)
        self.assertTrue(result.accepted_all)
        self.assertEqual(result.final_token_id, 2)
        self.assertEqual(result.token_ids, [0, 1, 2])

    def test_rejects_first_token_and_samples_correction(self):
        draft_token_ids = torch.tensor([0, 1])
        target_probs = torch.tensor([
            [0.10, 0.90, 0.00],
            [0.20, 0.60, 0.20],
            [0.20, 0.20, 0.60],
        ])
        draft_probs = torch.tensor([
            [0.90, 0.10, 0.00],
            [0.10, 0.80, 0.10],
        ])
        random_values = torch.tensor([0.5])

        result = speculative_accept_reject(
            target_probs,
            draft_probs,
            draft_token_ids,
            random_values=random_values,
        )

        self.assertEqual(result.accepted_token_ids, [])
        self.assertEqual(result.num_accepted, 0)
        self.assertFalse(result.accepted_all)
        self.assertEqual(result.final_token_id, 1)
        self.assertEqual(result.token_ids, [1])

    def test_rejects_after_accepting_prefix_and_samples_correction(self):
        draft_token_ids = torch.tensor([0, 1])
        target_probs = torch.tensor([
            [0.70, 0.20, 0.10],
            [0.90, 0.10, 0.00],
            [0.20, 0.20, 0.60],
        ])
        draft_probs = torch.tensor([
            [0.60, 0.30, 0.10],
            [0.10, 0.90, 0.00],
        ])
        random_values = torch.tensor([0.5, 0.5])

        result = speculative_accept_reject(
            target_probs,
            draft_probs,
            draft_token_ids,
            random_values=random_values,
        )

        self.assertEqual(result.accepted_token_ids, [0])
        self.assertEqual(result.num_accepted, 1)
        self.assertFalse(result.accepted_all)
        self.assertEqual(result.final_token_id, 0)
        self.assertEqual(result.token_ids, [0, 0])


if __name__ == "__main__":
    unittest.main()
