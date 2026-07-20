import unittest

import torch

from llmserve.models.eagle3 import Eagle3DraftOutput, run_eagle3_offline_step, run_eagle3_target_verify


class FakeTargetModel:

    def __init__(self):
        self.calls = []

    def forward_with_eagle3_aux(self, input_ids, positions):
        self.calls.append((input_ids.clone(), positions.clone()))
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 4)
        aux = torch.cat([hidden + 10, hidden + 20, hidden + 30], dim=-1)
        return hidden, aux

    def compute_logits(self, hidden_states):
        logits = torch.full((hidden_states.size(0), 8), -100.0)
        token_ids = hidden_states[:, 0].long()
        logits[torch.arange(hidden_states.size(0)), token_ids] = 0.0
        return logits


class FakeDraftModel:
    hidden_size = 4

    def __init__(self):
        self.tokens = [2, 3, 4]
        self.calls = []

    def propose(self, input_ids, aux_hidden_states, positions, temperature, past_kv=None, kv_valid_lens=None):
        step = len(self.calls)
        self.calls.append((int(input_ids.item()), aux_hidden_states.size(-1), int(positions.item())))
        hidden = torch.full((1, 1, self.hidden_size), float(step + 1))
        draft_logits = torch.zeros(1, 1, 8)
        target_logits = torch.full((1, 1, 8), -100.0)
        target_logits[0, 0, self.tokens[step]] = 0.0
        token_ids = torch.tensor([[self.tokens[step]]], dtype=torch.long)
        past = (torch.zeros(1, 1, step + 1, 1), torch.zeros(1, 1, step + 1, 1))
        return Eagle3DraftOutput(hidden, draft_logits, target_logits, token_ids, past)


class Eagle3TargetVerifyTest(unittest.TestCase):

    def test_runs_target_verify_for_start_and_draft_tokens(self):
        target = FakeTargetModel()

        result = run_eagle3_target_verify(
            target,
            start_token_id=1,
            draft_token_ids=[2, 3, 4],
            start_position=5,
        )

        self.assertEqual(result.target_logits.shape, (4, 8))
        self.assertEqual(result.target_aux_hidden.shape, (4, 12))
        self.assertEqual(result.target_logits.argmax(dim=-1).tolist(), [1, 2, 3, 4])
        self.assertEqual(target.calls[0][0].tolist(), [1, 2, 3, 4])
        self.assertEqual(target.calls[0][1].tolist(), [5, 6, 7, 8])

    def test_offline_step_runs_draft_verify_and_sampling(self):
        target = FakeTargetModel()
        draft = FakeDraftModel()

        result = run_eagle3_offline_step(
            target,
            draft,
            start_token_id=1,
            start_aux_hidden=torch.zeros(1, 1, 12),
            start_position=5,
            gamma=3,
            temperature=1.0,
            random_values=torch.zeros(3),
        )

        self.assertEqual(result.draft_token_ids, [2, 3, 4])
        self.assertEqual(result.verify_output.target_logits.argmax(dim=-1).tolist(), [1, 2, 3, 4])
        self.assertEqual(result.sample_result.accepted_token_ids, [2, 3, 4])
        self.assertEqual(result.sample_result.token_ids, [2, 3, 4, 4])
        self.assertEqual([call[1] for call in draft.calls], [12, 4, 4])


if __name__ == "__main__":
    unittest.main()
