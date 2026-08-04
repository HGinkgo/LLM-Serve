import unittest

from tests.test_request_abort import make_engine, make_scheduler
from llmserve.engine.sequence import Sequence, SequenceStatus


class RequestFailureTest(unittest.TestCase):
    def setUp(self):
        self.old_block_size = Sequence.block_size
        Sequence.block_size = 4

    def tearDown(self):
        Sequence.block_size = self.old_block_size

    def test_engine_failure_releases_resources_and_records_reason(self):
        class DraftRunner:
            draft_model = object()

            def __init__(self):
                self.calls = []

            def call(self, method, *args):
                self.calls.append((method, args))

        scheduler = make_scheduler()
        seq = Sequence([1, 2, 3, 4, 5])
        seq.status = SequenceStatus.RUNNING
        scheduler.running.append(seq)
        scheduler.block_manager.allocate(seq)
        scheduler._reserved_blocks[seq.seq_id] = 2
        runner = DraftRunner()
        engine = make_engine(scheduler, runner)
        engine.request_metrics[seq.seq_id] = engine._new_request_metric(seq)

        self.assertTrue(engine.fail_request(seq.seq_id, "decode worker failed"))
        self.assertFalse(engine.fail_request(seq.seq_id, "duplicate"))
        self.assertTrue(engine.is_finished())
        self.assertEqual(seq.status, SequenceStatus.FAILED)
        self.assertEqual(len(scheduler.block_manager.used_block_ids), 0)
        self.assertEqual(scheduler.kv_capacity_metrics()["reserved_blocks"], 0)
        self.assertEqual(
            runner.calls,
            [("clear_speculative_state", ([seq.seq_id],))],
        )

        metrics = engine.get_metrics()
        request = metrics["requests"][0]
        self.assertEqual(request["status"], "failed")
        self.assertFalse(request["cancelled"])
        self.assertEqual(request["failure_reason"], "decode worker failed")
        self.assertEqual(metrics["summary"]["num_finished"], 0)
        self.assertEqual(metrics["summary"]["num_cancelled"], 0)
        self.assertEqual(metrics["summary"]["num_failed"], 1)
        self.assertIsNone(metrics["summary"]["request_latency"]["mean"])


if __name__ == "__main__":
    unittest.main()
