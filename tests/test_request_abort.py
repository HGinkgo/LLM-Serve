import unittest
from collections import deque
from types import SimpleNamespace

from llmserve.engine.block_manager import BlockManager
from llmserve.engine.llm_engine import LLMEngine
from llmserve.engine.scheduler import Scheduler
from llmserve.engine.sequence import Sequence, SequenceStatus


class RequestAbortTest(unittest.TestCase):
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
        scheduler.enable_chunked_prefill = True
        scheduler.waiting = deque()
        scheduler.running = deque()
        scheduler.enable_kv_capacity_admission = True
        scheduler.max_model_len = 4096
        scheduler.speculative_reserve_tokens = 0
        scheduler._reserved_blocks = {}
        scheduler.preemption_count = 0
        scheduler.admission_deferred_count = 0
        scheduler.peak_reserved_blocks = 0
        return scheduler

    @staticmethod
    def make_engine(scheduler, model_runner=None):
        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = scheduler
        engine.model_runner = model_runner or SimpleNamespace(draft_model=None)
        engine.request_metrics = {}
        engine.last_step_events = {}
        engine.speculative_batch_calls = 0
        engine.speculative_batch_sequences = 0
        engine.speculative_max_batch_size = 0
        return engine

    def test_scheduler_aborts_waiting_sequence_and_releases_all_ownership(self):
        scheduler = self.make_scheduler()
        seq = Sequence([1, 2, 3, 4, 5])
        scheduler.waiting.append(seq)
        scheduler.block_manager.allocate(seq)
        scheduler._reserved_blocks[seq.seq_id] = 2

        aborted = scheduler.abort_request(seq.seq_id)

        self.assertIs(aborted, seq)
        self.assertEqual(seq.status, SequenceStatus.CANCELLED)
        self.assertEqual(list(scheduler.waiting), [])
        self.assertEqual(len(scheduler.block_manager.used_block_ids), 0)
        self.assertEqual(scheduler.kv_capacity_metrics()["reserved_blocks"], 0)

    def test_scheduler_aborts_running_sequence_and_releases_target_kv(self):
        scheduler = self.make_scheduler()
        seq = Sequence([1, 2, 3, 4, 5])
        seq.status = SequenceStatus.RUNNING
        seq.num_cached_tokens = len(seq)
        scheduler.running.append(seq)
        scheduler.block_manager.allocate(seq)

        aborted = scheduler.abort_request(seq.seq_id)

        self.assertIs(aborted, seq)
        self.assertEqual(seq.status, SequenceStatus.CANCELLED)
        self.assertEqual(list(scheduler.running), [])
        self.assertEqual(len(scheduler.block_manager.used_block_ids), 0)

    def test_engine_abort_is_idempotent_and_records_cancelled_outcome(self):
        scheduler = self.make_scheduler()
        seq = Sequence([1, 2, 3, 4])
        scheduler.waiting.append(seq)
        engine = self.make_engine(scheduler)
        metric = engine._new_request_metric(seq)
        seq.append_token(10)
        metric["first_token_time"] = metric["arrival_time"] + 0.001
        metric["token_times"] = [metric["first_token_time"]]
        metric["output_event_times"] = [metric["first_token_time"]]
        metric["output_tokens"] = 1
        engine.request_metrics[seq.seq_id] = metric

        self.assertTrue(engine.abort_request(seq.seq_id))
        self.assertFalse(engine.abort_request(seq.seq_id))
        self.assertFalse(engine.abort_request(seq.seq_id + 1000))
        self.assertTrue(engine.is_finished())

        metrics = engine.get_metrics()
        request = metrics["requests"][0]
        self.assertTrue(request["cancelled"])
        self.assertEqual(request["status"], "cancelled")
        self.assertFalse(request["success"])
        self.assertIsNone(request["failure_reason"])
        self.assertIsNotNone(request["finish_time"])
        self.assertEqual(metrics["summary"]["num_cancelled"], 1)
        self.assertEqual(metrics["summary"]["num_failed"], 0)
        self.assertIsNone(metrics["summary"]["request_latency"]["mean"])
        self.assertIsNone(metrics["summary"]["tpot"]["mean"])

    def test_engine_abort_clears_eagle_request_state(self):
        class DraftRunner:
            draft_model = object()

            def __init__(self):
                self.calls = []

            def call(self, method, *args):
                self.calls.append((method, args))

        scheduler = self.make_scheduler()
        seq = Sequence([1, 2, 3, 4])
        seq.status = SequenceStatus.RUNNING
        scheduler.running.append(seq)
        runner = DraftRunner()
        engine = self.make_engine(scheduler, runner)
        engine.request_metrics[seq.seq_id] = engine._new_request_metric(seq)

        self.assertTrue(engine.abort_request(seq.seq_id))
        self.assertEqual(
            runner.calls,
            [("clear_speculative_state", ([seq.seq_id],))],
        )


if __name__ == "__main__":
    unittest.main()
