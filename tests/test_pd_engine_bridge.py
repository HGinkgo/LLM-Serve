import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import torch

from llmserve.engine.block_manager import BlockManager
from llmserve.engine.llm_engine import LLMEngine
from llmserve.engine.scheduler import Scheduler
from llmserve.engine.sequence import Sequence, SequenceStatus
from llmserve.sampling_params import SamplingParams


class TestPDEngineBridge(unittest.TestCase):

    def setUp(self):
        self.old_block_size = Sequence.block_size
        Sequence.block_size = 4

    def tearDown(self):
        Sequence.block_size = self.old_block_size

    @staticmethod
    def make_scheduler():
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.eos = 99
        scheduler.block_manager = BlockManager(num_blocks=8, block_size=4)
        scheduler.max_num_seqs = 4
        scheduler.max_num_batched_tokens = 8
        scheduler.enable_chunked_prefill = False
        scheduler.enable_kv_capacity_admission = False
        scheduler.max_model_len = 4096
        scheduler.speculative_reserve_tokens = 0
        scheduler._reserved_blocks = {}
        scheduler.preemption_count = 0
        scheduler.admission_deferred_count = 0
        scheduler.peak_reserved_blocks = 0
        scheduler.waiting = deque()
        scheduler.running = deque()
        scheduler.pending_prefilled = deque()
        return scheduler

    def test_add_prefilled_request_imports_prompt_kv_and_records_first_token(self):
        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = self.make_scheduler()
        engine.request_metrics = {}
        engine.model_runner = SimpleNamespace(
            kv_cache=torch.zeros(2, 1, 8, 4, 1, 2, dtype=torch.float32)
        )
        prompt = [10, 11, 12, 13]
        payload = torch.full((2, 1, 4, 1, 2), 3.5)

        seq_id = engine.add_prefilled_request(
            prompt,
            first_token_id=77,
            sampling_params=SamplingParams(
                temperature=1.0,
                max_tokens=4,
                ignore_eos=True,
            ),
            kv_payload=payload,
        )

        seq = engine.scheduler.running[0]
        self.assertEqual(seq.seq_id, seq_id)
        self.assertEqual(seq.prompt_token_ids, prompt)
        self.assertEqual(seq.completion_token_ids, [77])
        self.assertEqual(seq.num_cached_tokens, len(prompt))
        self.assertEqual(seq.status, SequenceStatus.RUNNING)
        self.assertTrue(torch.equal(engine.model_runner.kv_cache[:, :, 0, :4], payload))
        self.assertEqual(engine.request_metrics[seq_id]["output_tokens"], 1)
        self.assertEqual(engine.request_metrics[seq_id]["prompt_tokens"], len(prompt))

    def test_kv_import_failure_releases_decode_worker_sequence(self):
        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = self.make_scheduler()
        engine.request_metrics = {}
        engine.model_runner = SimpleNamespace(
            kv_cache=torch.zeros(2, 1, 8, 4, 1, 2, dtype=torch.float32)
        )

        with self.assertRaises(ValueError):
            engine.add_prefilled_request(
                [10, 11, 12, 13],
                first_token_id=77,
                sampling_params=SamplingParams(
                    temperature=1.0,
                    max_tokens=4,
                    ignore_eos=True,
                ),
                kv_payload=torch.zeros(2, 2, 4, 1, 2),
            )

        self.assertEqual(list(engine.scheduler.running), [])
        self.assertEqual(len(engine.scheduler.block_manager.used_block_ids), 0)

    def test_async_prefilled_request_waits_for_event_before_decode_activation(self):
        class Completion:

            def __init__(self):
                self.ready = False

            def is_complete(self):
                return self.ready

        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = self.make_scheduler()
        engine.request_metrics = {}
        engine._prefilled_import_completions = {}
        engine.model_runner = SimpleNamespace(
            kv_cache=torch.zeros(2, 1, 8, 4, 1, 2, dtype=torch.float32)
        )
        completion = Completion()

        with patch(
            "llmserve.engine.llm_engine.import_logical_kv",
            return_value=completion,
        ):
            with patch.object(
                engine,
                "_prefilled_import_stream_for_kv_cache",
                return_value=object(),
            ):
                seq_id = engine.add_prefilled_request(
                    [10, 11, 12, 13],
                    first_token_id=77,
                    sampling_params=SamplingParams(
                        temperature=1.0,
                        max_tokens=4,
                        ignore_eos=True,
                    ),
                    kv_payload=torch.zeros(2, 1, 4, 1, 2),
                    async_kv_import=True,
                )

        self.assertEqual(list(engine.scheduler.running), [])
        self.assertEqual(
            [seq.seq_id for seq in engine.scheduler.pending_prefilled],
            [seq_id],
        )

        completion.ready = True
        self.assertEqual(engine.activate_completed_prefilled_imports(), [seq_id])
        self.assertEqual(
            [seq.seq_id for seq in engine.scheduler.running],
            [seq_id],
        )

    def test_async_import_abort_defers_target_block_release_until_event_completes(self):
        class Completion:

            def __init__(self):
                self.ready = False

            def is_complete(self):
                return self.ready

        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = self.make_scheduler()
        engine.request_metrics = {}
        engine._prefilled_import_completions = {}
        engine._prefilled_import_deferred_releases = {}
        engine.model_runner = SimpleNamespace(
            kv_cache=torch.zeros(2, 1, 8, 4, 1, 2, dtype=torch.float32),
            draft_model=None,
        )
        completion = Completion()

        with patch(
            "llmserve.engine.llm_engine.import_logical_kv",
            return_value=completion,
        ):
            with patch.object(
                engine,
                "_prefilled_import_stream_for_kv_cache",
                return_value=object(),
            ):
                seq_id = engine.add_prefilled_request(
                    [10, 11, 12, 13],
                    first_token_id=77,
                    sampling_params=SamplingParams(
                        temperature=1.0,
                        max_tokens=4,
                        ignore_eos=True,
                    ),
                    kv_payload=torch.zeros(2, 1, 4, 1, 2),
                    async_kv_import=True,
                )

        self.assertTrue(engine.abort_request(seq_id))
        self.assertEqual(len(engine.scheduler.block_manager.used_block_ids), 2)
        self.assertEqual(list(engine.scheduler.pending_prefilled), [])

        completion.ready = True
        engine.complete_prefilled_import(seq_id)
        self.assertEqual(len(engine.scheduler.block_manager.used_block_ids), 0)

    def test_terminal_first_token_defers_target_block_release_until_event_completes(self):
        class Completion:

            def __init__(self):
                self.ready = False

            def is_complete(self):
                return self.ready

        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = self.make_scheduler()
        engine.request_metrics = {}
        engine._prefilled_import_completions = {}
        engine._prefilled_import_deferred_releases = {}
        engine.model_runner = SimpleNamespace(
            kv_cache=torch.zeros(2, 1, 8, 4, 1, 2, dtype=torch.float32),
        )
        completion = Completion()

        with patch(
            "llmserve.engine.llm_engine.import_logical_kv",
            return_value=completion,
        ):
            with patch.object(
                engine,
                "_prefilled_import_stream_for_kv_cache",
                return_value=object(),
            ):
                seq_id = engine.add_prefilled_request(
                    [10, 11, 12, 13],
                    first_token_id=77,
                    sampling_params=SamplingParams(
                        temperature=1.0,
                        max_tokens=1,
                        ignore_eos=True,
                    ),
                    kv_payload=torch.zeros(2, 1, 4, 1, 2),
                    async_kv_import=True,
                )

        self.assertEqual(len(engine.scheduler.block_manager.used_block_ids), 2)
        self.assertEqual(list(engine.scheduler.pending_prefilled), [])

        completion.ready = True
        engine.complete_prefilled_import(seq_id)
        self.assertEqual(len(engine.scheduler.block_manager.used_block_ids), 0)

    def test_latency_telemetry_records_default_stream_decode_interval(self):
        class FakeEvent:

            def __init__(self):
                self.record_calls = 0

            def record(self):
                self.record_calls += 1

        engine = LLMEngine.__new__(LLMEngine)
        engine.config = SimpleNamespace(enable_latency_telemetry=True)
        engine.model_runner = SimpleNamespace(
            kv_cache=SimpleNamespace(device=torch.device("cuda:0"))
        )
        start_event = FakeEvent()
        end_event = FakeEvent()

        with patch(
            "llmserve.engine.llm_engine.torch.cuda.Event",
            side_effect=[start_event, end_event],
        ):
            interval = engine.begin_step_cuda_interval()
            engine.finish_step_cuda_interval(interval)

        self.assertIs(engine.last_step_cuda_interval, interval)
        self.assertEqual(start_event.record_calls, 1)
        self.assertEqual(end_event.record_calls, 1)


if __name__ == "__main__":
    unittest.main()
