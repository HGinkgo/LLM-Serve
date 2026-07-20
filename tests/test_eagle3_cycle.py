import unittest

import torch

from llmserve.models.eagle3 import (
    Eagle3DraftOutput,
    Eagle3Speculator,
    generate_eagle3_draft_tokens,
    generate_eagle3_draft_tokens_batched,
    run_eagle3_speculative_cycle,
    speculative_accept_greedy_from_logits,
)
from tests.support import (
    SPECULATIVE_MODEL_PATH,
    TARGET_MODEL_PATH,
    requires_eagle3_models,
)


class FakeDraftModel:
    hidden_size = 4
    target_vocab_size = 6

    def __init__(self, token_ids, draft_probs):
        self.token_ids = token_ids
        self.draft_logits = draft_probs.log()
        self.calls = []

    def propose(self, input_ids, aux_hidden_states, positions, temperature, past_kv=None, kv_valid_lens=None):
        step = len(self.calls)
        self.calls.append({
            "input_id": int(input_ids.item()),
            "aux_width": aux_hidden_states.size(-1),
            "position": int(positions.item()),
            "has_past": past_kv is not None,
        })

        hidden_states = torch.full((1, 1, self.hidden_size), float(step + 1))
        draft_logits = torch.zeros(1, 1, 3)
        target_logits = self.draft_logits[step].view(1, 1, -1)
        token_ids = torch.tensor([[self.token_ids[step]]], dtype=torch.long)
        past = (
            torch.zeros(1, 1, step + 1, 1),
            torch.zeros(1, 1, step + 1, 1),
        )
        return Eagle3DraftOutput(hidden_states, draft_logits, target_logits, token_ids, past)


class FakeGreedyDraftModel:
    hidden_size = 4

    def __init__(self):
        self.sample_token_ids = [5, 5, 5]
        self.greedy_token_ids = [1, 2, 3]
        self.calls = []

    def greedy_sample(self, draft_logits):
        return draft_logits.argmax(dim=-1)

    def propose(self, input_ids, aux_hidden_states, positions, temperature, past_kv=None, kv_valid_lens=None):
        step = len(self.calls)
        self.calls.append(int(input_ids.item()))
        hidden_states = torch.full((1, 1, self.hidden_size), float(step + 1))
        draft_logits = torch.full((1, 1, 8), -100.0)
        draft_logits[0, 0, self.greedy_token_ids[step]] = 0.0
        target_logits = torch.full((1, 1, 8), -100.0)
        target_logits[0, 0, self.greedy_token_ids[step]] = 0.0
        token_ids = torch.tensor([[self.sample_token_ids[step]]], dtype=torch.long)
        past = (
            torch.zeros(1, 1, step + 1, 1),
            torch.zeros(1, 1, step + 1, 1),
        )
        return Eagle3DraftOutput(hidden_states, draft_logits, target_logits, token_ids, past)


class FakeBatchedGreedyDraftModel:
    hidden_size = 4

    def __init__(self):
        self.calls = []

    def greedy_sample(self, draft_logits):
        return draft_logits.argmax(dim=-1)

    def propose(self, input_ids, aux_hidden_states, positions, temperature, past_kv=None, kv_valid_lens=None):
        batch_size = input_ids.size(0)
        self.calls.append({
            "input_ids": input_ids.flatten().tolist(),
            "positions": positions.flatten().tolist(),
            "past_len": 0 if past_kv is None else past_kv[0].size(2),
            "kv_valid_lens": None if kv_valid_lens is None else kv_valid_lens.tolist(),
        })
        next_tokens = input_ids + 1
        hidden_states = torch.ones(batch_size, 1, self.hidden_size)
        draft_logits = torch.full((batch_size, 1, 16), -100.0)
        draft_logits.scatter_(2, next_tokens.unsqueeze(-1), 0.0)
        target_logits = draft_logits.clone()
        if past_kv is None:
            past_k = torch.empty(batch_size, 1, 0, 1)
            past_v = torch.empty(batch_size, 1, 0, 1)
        else:
            past_k, past_v = past_kv
        new_value = positions.float().view(batch_size, 1, 1, 1)
        new_past = (
            torch.cat([past_k, new_value], dim=2),
            torch.cat([past_v, new_value + 100], dim=2),
        )
        return Eagle3DraftOutput(hidden_states, draft_logits, target_logits, next_tokens, new_past)


class Eagle3CycleTest(unittest.TestCase):

    def test_greedy_accepts_continuous_prefix_and_bonus(self):
        target_logits = torch.full((4, 8), -100.0)
        target_logits[0, 2] = 0.0
        target_logits[1, 3] = 0.0
        target_logits[2, 4] = 0.0
        target_logits[3, 7] = 0.0

        result = speculative_accept_greedy_from_logits(
            target_logits,
            torch.tensor([2, 3, 4]),
        )

        self.assertEqual(result.accepted_token_ids, [2, 3, 4])
        self.assertEqual(result.token_ids, [2, 3, 4, 7])
        self.assertEqual(result.final_token_id, 7)
        self.assertEqual(result.num_accepted, 3)
        self.assertTrue(result.accepted_all)

    def test_greedy_rejects_at_first_mismatch_and_uses_target_correction(self):
        target_logits = torch.full((4, 8), -100.0)
        target_logits[0, 2] = 0.0
        target_logits[1, 6] = 0.0
        target_logits[2, 4] = 0.0
        target_logits[3, 7] = 0.0

        result = speculative_accept_greedy_from_logits(
            target_logits,
            torch.tensor([2, 3, 4]),
        )

        self.assertEqual(result.accepted_token_ids, [2])
        self.assertEqual(result.token_ids, [2, 6])
        self.assertEqual(result.final_token_id, 6)
        self.assertEqual(result.num_accepted, 1)
        self.assertFalse(result.accepted_all)

    def test_generates_draft_tokens_without_target_verify(self):
        draft = FakeDraftModel(
            token_ids=[1, 2, 3],
            draft_probs=torch.tensor([
                [0.05, 0.80, 0.05, 0.05, 0.05, 0.00],
                [0.05, 0.05, 0.80, 0.05, 0.05, 0.00],
                [0.05, 0.05, 0.05, 0.80, 0.05, 0.00],
            ]),
        )

        result = generate_eagle3_draft_tokens(
            draft,
            start_token_id=0,
            start_aux_hidden=torch.zeros(1, 1, 12),
            start_position=7,
            gamma=3,
            temperature=1.0,
        )

        self.assertEqual(result.draft_token_ids, [1, 2, 3])
        self.assertEqual(result.draft_target_logits.shape, (3, 6))
        self.assertEqual(result.past_kv[0].shape[2], 3)

    def test_generates_draft_tokens_with_greedy_argmax_mode(self):
        draft = FakeGreedyDraftModel()

        result = generate_eagle3_draft_tokens(
            draft,
            start_token_id=0,
            start_aux_hidden=torch.zeros(1, 1, 12),
            start_position=7,
            gamma=3,
            temperature=1.0,
            draft_sampling_mode="greedy",
        )

        self.assertEqual(result.draft_token_ids, [1, 2, 3])
        self.assertEqual(draft.calls, [0, 1, 2])

    def test_batched_draft_matches_serial_with_different_kv_lengths(self):
        past1 = (
            torch.tensor([[[[10.0], [11.0]]]]),
            torch.tensor([[[[110.0], [111.0]]]]),
        )
        past2 = (
            torch.tensor([[[[20.0], [21.0], [22.0], [23.0]]]]),
            torch.tensor([[[[120.0], [121.0], [122.0], [123.0]]]]),
        )
        batched_model = FakeBatchedGreedyDraftModel()

        batched = generate_eagle3_draft_tokens_batched(
            batched_model,
            start_token_ids=[1, 5],
            start_aux_hidden=torch.zeros(2, 1, 12),
            start_positions=[2, 4],
            gamma=3,
            temperature=1.0,
            past_kv=[past1, past2],
            draft_sampling_mode="greedy",
        )

        serial = []
        for start_token_id, start_position, past in [(1, 2, past1), (5, 4, past2)]:
            serial.append(generate_eagle3_draft_tokens(
                FakeBatchedGreedyDraftModel(),
                start_token_id=start_token_id,
                start_aux_hidden=torch.zeros(1, 1, 12),
                start_position=start_position,
                gamma=3,
                temperature=1.0,
                past_kv=past,
                kv_valid_len=past[0].size(2),
                draft_sampling_mode="greedy",
            ))

        self.assertEqual([item.draft_token_ids for item in batched], [[2, 3, 4], [6, 7, 8]])
        self.assertEqual(
            [item.draft_token_ids for item in batched],
            [item.draft_token_ids for item in serial],
        )
        self.assertEqual(len(batched_model.calls), 3)
        self.assertEqual(batched_model.calls[0]["past_len"], 4)
        self.assertEqual(batched_model.calls[0]["kv_valid_lens"], [2, 4])
        self.assertEqual(batched_model.calls[1]["kv_valid_lens"], [3, 5])
        self.assertEqual(batched[0].past_kv[0].shape[2], 5)
        self.assertEqual(batched[1].past_kv[0].shape[2], 7)
        self.assertEqual(batched[0].past_kv[0][0, 0, :, 0].tolist(), [10.0, 11.0, 2.0, 3.0, 4.0])
        self.assertEqual(batched[1].past_kv[0][0, 0, :, 0].tolist(), [20.0, 21.0, 22.0, 23.0, 4.0, 5.0, 6.0])

    def test_batched_draft_supports_per_request_gamma(self):
        past1 = (
            torch.tensor([[[[10.0], [11.0]]]]),
            torch.tensor([[[[110.0], [111.0]]]]),
        )
        past2 = (
            torch.tensor([[[[20.0], [21.0], [22.0], [23.0]]]]),
            torch.tensor([[[[120.0], [121.0], [122.0], [123.0]]]]),
        )
        model = FakeBatchedGreedyDraftModel()

        result = generate_eagle3_draft_tokens_batched(
            model,
            start_token_ids=[1, 5],
            start_aux_hidden=torch.zeros(2, 1, 12),
            start_positions=[2, 4],
            gamma=3,
            temperature=1.0,
            past_kv=[past1, past2],
            draft_sampling_mode="greedy",
            gammas=[1, 3],
        )

        self.assertEqual([item.draft_token_ids for item in result], [[2], [6, 7, 8]])
        self.assertEqual([call["input_ids"] for call in model.calls], [[1, 5], [6], [7]])
        self.assertEqual([call["positions"] for call in model.calls], [[2, 4], [5], [6]])
        self.assertEqual(model.calls[1]["kv_valid_lens"], [5])
        self.assertEqual(result[0].past_kv[0].shape[2], 3)
        self.assertEqual(result[1].past_kv[0].shape[2], 7)

    def test_serial_draft_reports_proposal_stage_timings(self):
        model = FakeBatchedGreedyDraftModel()

        result = generate_eagle3_draft_tokens(
            model,
            start_token_id=1,
            start_aux_hidden=torch.zeros(1, 1, 12),
            start_position=0,
            gamma=2,
            temperature=1.0,
            draft_sampling_mode="greedy",
        )

        self.assertEqual(
            set(result.proposal_timing),
            {
                "draft_pack_time",
                "draft_forward_time",
                "draft_sample_time",
                "draft_compact_time",
            },
        )
        self.assertTrue(all(value >= 0 for value in result.proposal_timing.values()))

    def test_batched_draft_reports_per_request_proposal_stage_timings(self):
        model = FakeBatchedGreedyDraftModel()

        results = generate_eagle3_draft_tokens_batched(
            model,
            start_token_ids=[1, 5],
            start_aux_hidden=torch.zeros(2, 1, 12),
            start_positions=[0, 0],
            gamma=2,
            temperature=1.0,
            draft_sampling_mode="greedy",
        )

        self.assertEqual(len(results), 2)
        for result in results:
            self.assertEqual(
                set(result.proposal_timing),
                {
                    "draft_pack_time",
                    "draft_forward_time",
                    "draft_sample_time",
                    "draft_compact_time",
                },
            )
            self.assertTrue(all(value >= 0 for value in result.proposal_timing.values()))

    def test_runs_three_draft_steps_then_accepts_all_with_bonus(self):
        draft = FakeDraftModel(
            token_ids=[1, 2, 3],
            draft_probs=torch.tensor([
                [0.05, 0.80, 0.05, 0.05, 0.05, 0.00],
                [0.05, 0.05, 0.80, 0.05, 0.05, 0.00],
                [0.05, 0.05, 0.05, 0.80, 0.05, 0.00],
            ]),
        )
        target_verify_probs = torch.tensor([
            [1 / 6] * 6,
            [1 / 6] * 6,
            [1 / 6] * 6,
            [0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
        ])

        result = run_eagle3_speculative_cycle(
            draft,
            start_token_id=0,
            start_aux_hidden=torch.zeros(1, 1, 12),
            start_position=7,
            target_verify_logits=target_verify_probs.log(),
            gamma=3,
            temperature=1.0,
            random_values=torch.zeros(3),
        )

        self.assertEqual(result.draft_token_ids, [1, 2, 3])
        self.assertEqual(result.sample_result.token_ids, [1, 2, 3, 5])
        self.assertTrue(result.sample_result.accepted_all)
        self.assertEqual(result.draft_target_logits.shape, (3, 6))
        self.assertEqual([call["input_id"] for call in draft.calls], [0, 1, 2])
        self.assertEqual([call["aux_width"] for call in draft.calls], [12, 4, 4])
        self.assertEqual([call["position"] for call in draft.calls], [7, 8, 9])
        self.assertEqual([call["has_past"] for call in draft.calls], [False, True, True])

    def test_rejects_after_accepted_prefix(self):
        draft = FakeDraftModel(
            token_ids=[1, 2, 3],
            draft_probs=torch.tensor([
                [0.05, 0.80, 0.05, 0.05, 0.05, 0.00],
                [0.05, 0.05, 0.80, 0.05, 0.05, 0.00],
                [0.05, 0.05, 0.05, 0.80, 0.05, 0.00],
            ]),
        )
        target_verify_probs = torch.tensor([
            [0.05, 0.90, 0.05, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.10, 0.00, 0.90, 0.00],
            [1 / 6] * 6,
            [1 / 6] * 6,
        ])

        result = run_eagle3_speculative_cycle(
            draft,
            start_token_id=0,
            start_aux_hidden=torch.zeros(1, 1, 12),
            start_position=0,
            target_verify_logits=target_verify_probs.log(),
            gamma=3,
            temperature=1.0,
            random_values=torch.tensor([0.0, 0.9, 0.0]),
        )

        self.assertEqual(result.sample_result.accepted_token_ids, [1])
        self.assertEqual(result.sample_result.token_ids, [1, 4])
        self.assertFalse(result.sample_result.accepted_all)

    @requires_eagle3_models
    def test_real_qwen3_draft_runs_three_step_cycle(self):
        draft = Eagle3Speculator.from_pretrained(
            SPECULATIVE_MODEL_PATH,
            target_model_path=TARGET_MODEL_PATH,
        ).float().eval()
        target_verify_logits = torch.zeros(4, draft.target_vocab_size)
        target_verify_logits[-1].fill_(float("-inf"))
        target_verify_logits[-1, 151645] = 0.0

        torch.manual_seed(0)
        with torch.inference_mode():
            result = run_eagle3_speculative_cycle(
                draft,
                start_token_id=151644,
            start_aux_hidden=torch.zeros(1, 1, 3 * draft.target_hidden_size),
                start_position=0,
                target_verify_logits=target_verify_logits,
                gamma=3,
                temperature=1.0,
                random_values=torch.zeros(3),
            )

        self.assertEqual(len(result.draft_token_ids), 3)
        self.assertEqual(result.draft_target_logits.shape, (3, draft.target_vocab_size))
        self.assertTrue(result.sample_result.accepted_all)
        self.assertEqual(result.sample_result.final_token_id, 151645)
        self.assertEqual(result.past_kv[0].shape[2], 3)


if __name__ == "__main__":
    unittest.main()
