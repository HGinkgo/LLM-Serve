import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from llmserve.engine.model_runner import ModelRunner
from llmserve.engine.sequence import Sequence
from llmserve.models.eagle3 import (
    Eagle3DraftOutput,
    Eagle3DraftSequence,
    Eagle3Speculator,
    Eagle3TargetVerifyOutput,
    SpeculativeSampleResult,
)
from tests.support import (
    SPECULATIVE_MODEL_PATH,
    TARGET_MODEL_PATH,
    requires_eagle3_models,
)


class FakeTargetModel:

    def __init__(self):
        self.forward_calls = []

    def forward_with_eagle3_aux(self, input_ids, positions):
        self.forward_calls.append((input_ids.clone(), positions.clone()))
        hidden = torch.ones(input_ids.numel(), 4)
        aux = torch.ones(input_ids.numel(), 12)
        return hidden, aux

    def compute_logits(self, hidden_states, all_tokens=False):
        logits = torch.zeros(hidden_states.size(0), 8)
        logits[:, 6] = 10
        return logits


class FakeSampler:

    def __call__(self, logits, temperatures):
        return torch.tensor([6], dtype=torch.long)


class FakeDraftModel:
    hidden_size = 4

    def __init__(self):
        self.tokens = [11, 12, 13, 99, 98, 97]
        self.calls = []
        self.forward_calls = []

    def greedy_sample(self, draft_logits):
        return draft_logits.argmax(dim=-1)

    def propose(self, input_ids, aux_hidden_states, positions, temperature, past_kv=None, kv_valid_lens=None):
        step = len(self.calls)
        aux_marker = float(aux_hidden_states.reshape(-1)[0].item())
        self.calls.append((
            int(input_ids.item()),
            aux_hidden_states.size(-1),
            int(positions.item()),
            past_kv is not None,
            None if kv_valid_lens is None else int(kv_valid_lens.item()),
            aux_marker,
        ))
        hidden = torch.full((1, 1, self.hidden_size), float(step + 1))
        draft_logits = torch.full((1, 1, 16), -100.0)
        draft_logits[0, 0, self.tokens[step]] = 0.0
        target_logits = torch.full((1, 1, 16), -100.0)
        target_logits[0, 0, self.tokens[step]] = 0.0
        token_ids = torch.tensor([[self.tokens[step]]], dtype=torch.long)
        past_len = 0 if past_kv is None else past_kv[0].shape[2]
        past = (torch.zeros(1, 1, past_len + 1, 1), torch.zeros(1, 1, past_len + 1, 1))
        return Eagle3DraftOutput(hidden, draft_logits, target_logits, token_ids, past)

    def __call__(self, input_ids, aux_hidden_states, positions, past_kv=None, kv_valid_lens=None):
        if past_kv is not None:
            aux_marker = float(aux_hidden_states.reshape(-1)[0].item())
            self.calls.append((
                int(input_ids.item()),
                aux_hidden_states.size(-1),
                int(positions.item()),
                True,
                None if kv_valid_lens is None else int(kv_valid_lens.item()),
                aux_marker,
            ))
        self.forward_calls.append((
            input_ids.tolist(),
            tuple(aux_hidden_states.shape),
            positions.tolist(),
            past_kv is not None,
        ))
        seq_len = input_ids.size(1)
        past_len = 0 if past_kv is None else past_kv[0].shape[2]
        past = (torch.zeros(1, 1, past_len + seq_len, 1), torch.zeros(1, 1, past_len + seq_len, 1))
        return torch.zeros(1, seq_len, self.hidden_size), torch.zeros(1, seq_len, 16), past


class ModelRunnerSpeculativeTest(unittest.TestCase):

    def make_aux_hidden(self, markers):
        aux = torch.zeros(len(markers), 12)
        for i, marker in enumerate(markers):
            aux[i, 0] = marker
        return aux

    def test_load_draft_model_returns_none_without_speculative_model(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.config = SimpleNamespace(speculative_model=None, model="/unused")

        self.assertIsNone(ModelRunner.load_draft_model(runner))

    @requires_eagle3_models
    def test_load_draft_model_loads_qwen3_eagle3_speculator(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.config = SimpleNamespace(
            model=TARGET_MODEL_PATH,
            speculative_model=SPECULATIVE_MODEL_PATH,
        )

        draft_model = ModelRunner.load_draft_model(runner)

        self.assertIsInstance(draft_model, Eagle3Speculator)
        self.assertFalse(draft_model.training)
        self.assertEqual(draft_model.hidden_size, 4096)
        self.assertEqual(draft_model.target_hidden_size, 4096)
        self.assertEqual(draft_model.draft_vocab_size, 32000)
        self.assertEqual(draft_model.target_vocab_size, 151936)

    def test_clear_speculative_state_releases_finished_request_state(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.draft_kv_cache = {1: "kv1", 2: "kv2"}
        runner._prefill_aux_chunks = {1: ["aux1"], 2: ["aux2"]}
        runner._prev_correction = {1: (10, "hidden1"), 2: (20, "hidden2")}

        ModelRunner.clear_speculative_state(runner, [1])

        self.assertNotIn(1, runner.draft_kv_cache)
        self.assertNotIn(1, runner._prefill_aux_chunks)
        self.assertNotIn(1, runner._prev_correction)
        self.assertIn(2, runner.draft_kv_cache)
        self.assertIn(2, runner._prefill_aux_chunks)
        self.assertIn(2, runner._prev_correction)

    def test_run_target_decode_with_eagle3_aux_samples_start_token(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.model = FakeTargetModel()
        runner.sampler = FakeSampler()
        runner.prepare_decode = lambda seqs: (
            torch.tensor([11], dtype=torch.long),
            torch.tensor([4], dtype=torch.long),
        )
        runner.prepare_sample = lambda seqs: torch.tensor([1.0])

        output = ModelRunner.run_target_decode_with_eagle3_aux(runner, [object()])

        self.assertEqual(output.token_ids, [6])
        self.assertEqual(output.positions.tolist(), [4])
        self.assertEqual(output.aux_hidden.shape, (1, 12))
        self.assertEqual(output.logits.shape, (1, 8))
        self.assertEqual(runner.model.forward_calls[0][0].tolist(), [11])
        self.assertEqual(runner.model.forward_calls[0][1].tolist(), [4])

    def test_build_target_verify_batch_metadata_packs_multiple_sequences(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.block_size = 4
        seq1 = Sequence([1, 2, 3, 4, 5])
        seq2 = Sequence([6, 7, 8, 9, 10, 11])
        seq1.block_table = [10, 11]
        seq2.block_table = [20, 21]

        metadata = ModelRunner._build_target_verify_batch_metadata(
            runner,
            [seq1, seq2],
            [30, 40],
            [[31, 32], [41, 42]],
            [0, -1],
        )

        self.assertEqual(metadata["input_ids"], [30, 31, 32, 40, 41, 42])
        self.assertEqual(metadata["positions"], [5, 6, 7, 5, 6, 7])
        self.assertEqual(metadata["slot_mapping"], [45, 46, 47, 85, 86, 87])
        self.assertEqual(metadata["cu_seqlens_q"], [0, 3, 6])
        self.assertEqual(metadata["cu_seqlens_k"], [0, 8, 16])
        self.assertEqual(metadata["max_seqlen_q"], 3)
        self.assertEqual(metadata["max_seqlen_k"], 8)
        self.assertEqual(metadata["verify_lengths"], [3, 3])

    def test_run_speculative_batch_uses_one_target_verify_for_two_sequences(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.draft_model = FakeDraftModel()
        runner.draft_model.tokens = [11, 12, 13, 7, 8, 9]
        runner.speculative_gamma = 3
        runner.speculative_accept_mode = "greedy"
        runner.speculative_trace = False
        runner.draft_kv_cache = {
            0: (torch.ones(1, 1, 5, 1), torch.ones(1, 1, 5, 1)),
            1: (torch.ones(1, 1, 5, 1), torch.ones(1, 1, 5, 1)),
        }
        runner._prev_correction = {}
        runner.run_target_decode_with_eagle3_aux = lambda seqs: SimpleNamespace(
            token_ids=[10, 6],
            aux_hidden=self.make_aux_hidden([50, 60]),
            positions=torch.tensor([4, 4]),
        )

        verify_calls = []
        target_logits1 = torch.full((4, 16), -100.0)
        target_logits1[0, 11] = 0.0
        target_logits1[1, 12] = 0.0
        target_logits1[2, 13] = 0.0
        target_logits1[3, 15] = 0.0
        target_logits2 = torch.full((4, 16), -100.0)
        target_logits2[0, 7] = 0.0
        target_logits2[1, 14] = 0.0
        target_logits2[2, 9] = 0.0
        target_logits2[3, 15] = 0.0

        def verify(seqs, start_tokens, draft_tokens, base_offsets):
            verify_calls.append((seqs, start_tokens, draft_tokens, base_offsets))
            return [
                Eagle3TargetVerifyOutput(target_logits1, self.make_aux_hidden([100, 101, 102, 103])),
                Eagle3TargetVerifyOutput(target_logits2, self.make_aux_hidden([200, 201, 202, 203])),
            ]

        runner.run_target_verify_batch_with_eagle3_aux = verify
        seq1 = Sequence([1, 2, 3, 4, 5])
        seq2 = Sequence([6, 7, 8, 9, 10])
        seq1.seq_id = 0
        seq2.seq_id = 1

        draft_sequences = [
            Eagle3DraftSequence(
                [11, 12, 13],
                torch.zeros(3, 16),
                (torch.zeros(1, 1, 8, 1), torch.zeros(1, 1, 8, 1)),
                {
                    "draft_pack_time": 0.001,
                    "draft_forward_time": 0.002,
                    "draft_sample_time": 0.003,
                    "draft_compact_time": 0.004,
                },
            ),
            Eagle3DraftSequence(
                [7, 8, 9],
                torch.zeros(3, 16),
                (torch.zeros(1, 1, 8, 1), torch.zeros(1, 1, 8, 1)),
                {
                    "draft_pack_time": 0.005,
                    "draft_forward_time": 0.006,
                    "draft_sample_time": 0.007,
                    "draft_compact_time": 0.008,
                },
            ),
        ]
        with patch(
            "llmserve.engine.speculative_executor.generate_eagle3_draft_tokens_batched",
            return_value=draft_sequences,
        ) as generate_batch:
            outputs = ModelRunner.run_speculative_batch(runner, [seq1, seq2])

        self.assertEqual(generate_batch.call_count, 1)
        self.assertEqual(len(verify_calls), 1)
        self.assertEqual(verify_calls[0][1], [10, 6])
        self.assertEqual(verify_calls[0][2], [[11, 12, 13], [7, 8, 9]])
        self.assertEqual(outputs[0].token_ids, [10, 11, 12, 13, 15])
        self.assertEqual(outputs[0].num_accepted, 3)
        self.assertEqual(outputs[1].token_ids, [6, 7, 14])
        self.assertEqual(outputs[1].num_accepted, 1)
        self.assertEqual(outputs[0].timing["draft_pack_time"], 0.001)
        self.assertEqual(outputs[0].timing["draft_compact_time"], 0.004)
        self.assertEqual(outputs[1].timing["draft_forward_time"], 0.006)
        self.assertEqual(outputs[1].timing["draft_sample_time"], 0.007)
        self.assertEqual(runner._prev_correction[0][0], 15)
        self.assertEqual(runner._prev_correction[1][0], 14)

    def test_generate_speculative_drafts_uses_serial_path_for_rejection(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.draft_model = object()
        runner.speculative_gamma = 3
        states = [
            {
                "seq": SimpleNamespace(temperature=1.0),
                "start_token_id": 10,
                "start_aux_hidden": torch.zeros(1, 1, 12),
                "draft_kv_len": 5,
                "draft_past_kv": None,
            },
            {
                "seq": SimpleNamespace(temperature=1.0),
                "start_token_id": 20,
                "start_aux_hidden": torch.zeros(1, 1, 12),
                "draft_kv_len": 7,
                "draft_past_kv": None,
            },
        ]
        serial_results = [
            Eagle3DraftSequence([11, 12, 13], torch.zeros(3, 16), None),
            Eagle3DraftSequence([21, 22, 23], torch.zeros(3, 16), None),
        ]

        with patch(
            "llmserve.engine.speculative_executor.generate_eagle3_draft_tokens_batched",
            side_effect=AssertionError("batched draft must be disabled"),
        ), patch(
            "llmserve.engine.speculative_executor.generate_eagle3_draft_tokens",
            side_effect=serial_results,
        ) as generate_serial:
            ModelRunner._generate_speculative_draft_sequences(runner, states, "rejection")

        self.assertEqual(generate_serial.call_count, 2)
        self.assertEqual(states[0]["draft_sequence"].draft_token_ids, [11, 12, 13])
        self.assertEqual(states[1]["draft_sequence"].draft_token_ids, [21, 22, 23])

    def test_generate_speculative_drafts_passes_per_request_gammas_to_batch(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.draft_model = object()
        runner.speculative_gamma = 3
        states = [
            {
                "seq": SimpleNamespace(temperature=1.0),
                "start_token_id": 10,
                "start_aux_hidden": torch.zeros(1, 1, 12),
                "draft_kv_len": 5,
                "draft_past_kv": None,
                "gamma": 1,
            },
            {
                "seq": SimpleNamespace(temperature=1.0),
                "start_token_id": 20,
                "start_aux_hidden": torch.zeros(1, 1, 12),
                "draft_kv_len": 7,
                "draft_past_kv": None,
                "gamma": 3,
            },
        ]
        draft_sequences = [
            Eagle3DraftSequence([11], torch.zeros(1, 16), None),
            Eagle3DraftSequence([21, 22, 23], torch.zeros(3, 16), None),
        ]

        with patch(
            "llmserve.engine.speculative_executor.generate_eagle3_draft_tokens_batched",
            return_value=draft_sequences,
        ) as generate_batch:
            ModelRunner._generate_speculative_draft_sequences(runner, states, "greedy")

        self.assertEqual(generate_batch.call_args.kwargs["gammas"], [1, 3])
        self.assertEqual(states[0]["draft_sequence"].draft_token_ids, [11])
        self.assertEqual(states[1]["draft_sequence"].draft_token_ids, [21, 22, 23])

    def test_update_draft_kv_uses_actual_request_gamma(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.speculative_gamma = 3
        runner.draft_kv_cache = {}
        seq = Sequence([1, 2, 3, 4, 5])
        past = (
            torch.zeros(1, 1, 8, 1),
            torch.zeros(1, 1, 8, 1),
        )

        ModelRunner._update_single_draft_kv(
            runner,
            seq,
            past,
            old_len=5,
            emitted_token_ids=[6, 7],
            verify_aux_hidden=torch.zeros(2, 12),
            num_accepted=1,
            gamma=1,
        )

        self.assertEqual(runner.draft_kv_cache[seq.seq_id][0].shape[2], 6)

    def test_run_speculative_single_returns_start_token_and_accepted_draft_tokens(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.draft_model = FakeDraftModel()
        runner.speculative_gamma = 3
        runner.draft_kv_cache = {
            0: (
                torch.ones(1, 1, 5, 1),
                torch.ones(1, 1, 5, 1),
            )
        }

        runner.run_target_decode_with_eagle3_aux = lambda seqs: SimpleNamespace(
            token_ids=[10],
            aux_hidden=self.make_aux_hidden([50]),
            positions=torch.tensor([4]),
        )
        target_logits = torch.full((4, 16), -100.0)
        target_logits[0, 11] = 0.0
        target_logits[1, 12] = 0.0
        target_logits[2, 13] = 0.0
        target_logits[3, 15] = 0.0
        runner.run_target_verify_with_eagle3_aux = lambda seq, start, drafts, base_offset=0: Eagle3TargetVerifyOutput(
            target_logits=target_logits,
            target_aux_hidden=self.make_aux_hidden([100, 101, 102, 103]),
        )

        seq = Sequence([1, 2, 3, 4, 5])
        seq.seq_id = 0
        output = ModelRunner.run_speculative_single(runner, seq)

        self.assertEqual(output.token_ids, [10, 11, 12, 13, 15])
        self.assertEqual(output.num_draft_tokens, 3)
        self.assertEqual(output.num_accepted, 3)
        self.assertTrue(output.accepted_all)
        self.assertEqual(output.emitted_tokens, 5)
        self.assertIsNotNone(output.timing)
        self.assertGreaterEqual(output.timing["draft_proposal_time"], 0.0)
        for name in (
            "draft_pack_time",
            "draft_forward_time",
            "draft_sample_time",
            "draft_compact_time",
        ):
            self.assertGreaterEqual(output.timing[name], 0.0)
        self.assertGreaterEqual(output.timing["target_verify_time"], 0.0)
        self.assertGreaterEqual(output.timing["accept_time"], 0.0)
        self.assertGreaterEqual(output.timing["kv_update_time"], 0.0)
        self.assertGreaterEqual(output.timing["trace_time"], 0.0)
        self.assertGreaterEqual(output.timing["total_time"], 0.0)
        self.assertEqual(
            runner.draft_model.calls,
            [
                (10, 12, 5, True, 5, 50.0),
                (11, 4, 6, True, 6, 1.0),
                (12, 4, 7, True, 7, 2.0),
            ],
        )
        self.assertEqual(runner.draft_kv_cache[seq.seq_id][0].shape[2], 8)
        self.assertEqual(runner._prev_correction[seq.seq_id][0], 15)
        self.assertEqual(float(runner._prev_correction[seq.seq_id][1][0].item()), 103.0)

    def test_run_speculative_single_defaults_to_greedy_accept_mode(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.draft_model = FakeDraftModel()
        runner.speculative_gamma = 3
        runner.draft_kv_cache = {
            0: (
                torch.ones(1, 1, 5, 1),
                torch.ones(1, 1, 5, 1),
            )
        }
        runner.run_target_decode_with_eagle3_aux = lambda seqs: SimpleNamespace(
            token_ids=[10],
            aux_hidden=self.make_aux_hidden([50]),
            positions=torch.tensor([4]),
        )
        target_logits = torch.full((4, 16), -100.0)
        target_logits[0, 11] = 0.0
        target_logits[1, 12] = 0.0
        target_logits[2, 13] = 0.0
        target_logits[3, 15] = 0.0
        runner.run_target_verify_with_eagle3_aux = lambda seq, start, drafts, base_offset=0: Eagle3TargetVerifyOutput(
            target_logits=target_logits,
            target_aux_hidden=self.make_aux_hidden([100, 101, 102, 103]),
        )
        seq = Sequence([1, 2, 3, 4, 5])
        seq.seq_id = 0

        with patch(
            "llmserve.engine.speculative_executor.speculative_accept_reject_from_logits",
            side_effect=AssertionError("rejection helper should not be used by default"),
        ):
            output = ModelRunner.run_speculative_single(runner, seq)

        self.assertEqual(output.token_ids, [10, 11, 12, 13, 15])
        self.assertEqual(output.num_accepted, 3)
        self.assertTrue(output.accepted_all)

    def test_run_speculative_single_can_use_rejection_accept_mode(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.draft_model = FakeDraftModel()
        runner.speculative_gamma = 3
        runner.speculative_accept_mode = "rejection"
        runner.draft_kv_cache = {
            0: (
                torch.ones(1, 1, 5, 1),
                torch.ones(1, 1, 5, 1),
            )
        }
        runner.run_target_decode_with_eagle3_aux = lambda seqs: SimpleNamespace(
            token_ids=[10],
            aux_hidden=self.make_aux_hidden([50]),
            positions=torch.tensor([4]),
        )
        target_logits = torch.full((4, 16), -100.0)
        target_logits[0, 14] = 0.0
        target_logits[1, 12] = 0.0
        target_logits[2, 13] = 0.0
        target_logits[3, 15] = 0.0
        runner.run_target_verify_with_eagle3_aux = lambda seq, start, drafts, base_offset=0: Eagle3TargetVerifyOutput(
            target_logits=target_logits,
            target_aux_hidden=self.make_aux_hidden([200, 201, 202, 203]),
        )
        seq = Sequence([1, 2, 3, 4, 5])
        seq.seq_id = 0
        rejection_result = SpeculativeSampleResult(
            token_ids=[14],
            accepted_token_ids=[],
            final_token_id=14,
            num_accepted=0,
            accepted_all=False,
        )

        with patch(
            "llmserve.engine.speculative_executor.speculative_accept_reject_from_logits",
            return_value=rejection_result,
        ) as accept_reject:
            output = ModelRunner.run_speculative_single(runner, seq)

        self.assertEqual(output.token_ids, [10, 14])
        self.assertEqual(output.num_accepted, 0)
        self.assertFalse(output.accepted_all)
        self.assertEqual(accept_reject.call_count, 1)

    def test_run_speculative_single_can_emit_trace_debug(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.draft_model = FakeDraftModel()
        runner.speculative_gamma = 3
        runner.speculative_trace = True
        runner.draft_kv_cache = {
            0: (
                torch.ones(1, 1, 5, 1),
                torch.ones(1, 1, 5, 1),
            )
        }
        runner.run_target_decode_with_eagle3_aux = lambda seqs: SimpleNamespace(
            token_ids=[10],
            aux_hidden=self.make_aux_hidden([50]),
            positions=torch.tensor([4]),
        )
        target_logits = torch.full((4, 16), -100.0)
        target_logits[0, 11] = 0.0
        target_logits[1, 14] = 0.0
        target_logits[2, 13] = 0.0
        target_logits[3, 15] = 0.0
        runner.run_target_verify_with_eagle3_aux = lambda seq, start, drafts, base_offset=0: Eagle3TargetVerifyOutput(
            target_logits=target_logits,
            target_aux_hidden=self.make_aux_hidden([200, 201, 202, 203]),
        )

        seq = Sequence([1, 2, 3, 4, 5])
        seq.seq_id = 0
        output = ModelRunner.run_speculative_single(runner, seq)

        self.assertEqual(output.token_ids, [10, 11, 14])
        self.assertIsNotNone(output.debug)
        self.assertEqual(output.debug["accept_mode"], "greedy")
        self.assertEqual(output.debug["start_token_id"], 10)
        self.assertEqual(output.debug["draft_token_ids"], [11, 12, 13])
        self.assertEqual(output.debug["target_argmax_token_ids"], [11, 14, 13, 15])
        self.assertEqual(output.debug["matches"], [True, False, True])
        self.assertEqual(output.debug["draft_token_target_ranks"], [1, 2, 1])

    def test_run_speculative_single_saves_rejected_correction_for_merged_step(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.draft_model = FakeDraftModel()
        runner.speculative_gamma = 3
        runner.draft_kv_cache = {
            0: (
                torch.ones(1, 1, 5, 1),
                torch.ones(1, 1, 5, 1),
            )
        }

        runner.run_target_decode_with_eagle3_aux = lambda seqs: SimpleNamespace(
            token_ids=[10],
            aux_hidden=self.make_aux_hidden([50]),
            positions=torch.tensor([4]),
        )
        target_logits = torch.full((4, 16), -100.0)
        target_logits[0, 14] = 0.0
        target_logits[1, 12] = 0.0
        target_logits[2, 13] = 0.0
        target_logits[3, 15] = 0.0
        runner.run_target_verify_with_eagle3_aux = lambda seq, start, drafts, base_offset=0: Eagle3TargetVerifyOutput(
            target_logits=target_logits,
            target_aux_hidden=self.make_aux_hidden([200, 201, 202, 203]),
        )

        seq = Sequence([1, 2, 3, 4, 5])
        seq.seq_id = 0
        output = ModelRunner.run_speculative_single(runner, seq)

        self.assertEqual(output.token_ids, [10, 14])
        self.assertEqual(output.num_draft_tokens, 3)
        self.assertEqual(output.num_accepted, 0)
        self.assertFalse(output.accepted_all)
        self.assertEqual(output.emitted_tokens, 2)
        self.assertEqual(
            runner.draft_model.calls,
            [
                (10, 12, 5, True, 5, 50.0),
                (11, 4, 6, True, 6, 1.0),
                (12, 4, 7, True, 7, 2.0),
            ],
        )
        self.assertEqual(runner.draft_kv_cache[seq.seq_id][0].shape[2], 6)
        self.assertEqual(runner._prev_correction[seq.seq_id][0], 14)
        self.assertEqual(float(runner._prev_correction[seq.seq_id][1][0].item()), 200.0)

    def test_run_speculative_single_merged_step_reuses_saved_correction(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.draft_model = FakeDraftModel()
        runner.speculative_gamma = 3
        runner.draft_kv_cache = {
            0: (
                torch.ones(1, 1, 6, 1),
                torch.ones(1, 1, 6, 1),
            )
        }
        runner._prev_correction = {0: (14, self.make_aux_hidden([200])[0])}
        runner.run_target_decode_with_eagle3_aux = lambda seqs: (_ for _ in ()).throw(
            AssertionError("merged speculative step should skip target decode")
        )

        captured = {}
        target_logits = torch.full((4, 16), -100.0)
        target_logits[0, 11] = 0.0
        target_logits[1, 12] = 0.0
        target_logits[2, 13] = 0.0
        target_logits[3, 15] = 0.0

        def verify(seq, start, drafts, base_offset=0):
            captured["start"] = start
            captured["drafts"] = drafts
            captured["base_offset"] = base_offset
            return Eagle3TargetVerifyOutput(
                target_logits=target_logits,
                target_aux_hidden=self.make_aux_hidden([100, 101, 102, 103]),
            )

        runner.run_target_verify_with_eagle3_aux = verify

        seq = Sequence([1, 2, 3, 4, 5, 14])
        seq.seq_id = 0
        output = ModelRunner.run_speculative_single(runner, seq)

        self.assertEqual(output.token_ids, [11, 12, 13, 15])
        self.assertEqual(captured, {"start": 14, "drafts": [11, 12, 13], "base_offset": -1})
        self.assertEqual(
            runner.draft_model.calls,
            [
                (14, 12, 6, True, 6, 200.0),
                (11, 4, 7, True, 7, 1.0),
                (12, 4, 8, True, 8, 2.0),
            ],
        )
        self.assertEqual(runner.draft_kv_cache[seq.seq_id][0].shape[2], 9)
        self.assertEqual(runner._prev_correction[seq.seq_id][0], 15)

    def test_draft_prefill_stores_prompt_kv_cache(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.draft_model = FakeDraftModel()
        runner.draft_kv_cache = {}

        seq = Sequence([101, 102, 103])
        prompt_aux_hidden = self.make_aux_hidden([100, 101, 102])

        ModelRunner._draft_prefill(runner, seq, prompt_aux_hidden)

        self.assertIn(seq.seq_id, runner.draft_kv_cache)
        self.assertEqual(runner.draft_kv_cache[seq.seq_id][0].shape[2], 2)
        self.assertEqual(
            runner.draft_model.forward_calls,
            [([[102, 103]], (1, 2, 12), [[0, 1]], False)],
        )

    def test_prefill_sampled_token_is_appended_to_draft_kv(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.draft_model = FakeDraftModel()
        runner.draft_kv_cache = {
            0: (
                torch.ones(1, 1, 2, 1),
                torch.ones(1, 1, 2, 1),
            )
        }

        seq = Sequence([101, 102, 103])
        seq.seq_id = 0
        seq.num_cached_tokens = 0
        seq.num_scheduled_tokens = 3
        aux_hidden = self.make_aux_hidden([100, 101, 102])

        ModelRunner._fill_prefill_sampled_tokens(runner, [seq], aux_hidden, [10])

        self.assertEqual(runner.draft_kv_cache[seq.seq_id][0].shape[2], 3)
        self.assertEqual(
            runner.draft_model.calls,
            [(10, 12, 2, True, 2, 102.0)],
        )


if __name__ == "__main__":
    unittest.main()
