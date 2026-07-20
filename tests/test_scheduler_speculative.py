import unittest
from types import SimpleNamespace

from llmserve.engine.block_manager import BlockManager
from llmserve.engine.scheduler import Scheduler
from llmserve.engine.sequence import Sequence, SequenceStatus


class SchedulerSpeculativeTest(unittest.TestCase):

    def setUp(self):
        self.old_block_size = Sequence.block_size
        Sequence.block_size = 4

    def tearDown(self):
        Sequence.block_size = self.old_block_size

    def make_scheduler(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.eos = 99
        scheduler.block_manager = BlockManager(num_blocks=8, block_size=4)
        scheduler.running = []
        return scheduler

    def test_postprocess_speculative_appends_multiple_tokens_and_releases_extra_slots(self):
        scheduler = self.make_scheduler()
        seq = Sequence([1, 2, 3])
        seq.status = SequenceStatus.RUNNING
        seq.num_cached_tokens = len(seq)
        scheduler.running.append(seq)
        scheduler.block_manager.allocate(seq)
        scheduler.block_manager.ensure_slots(seq, num_tokens=5)

        Scheduler.postprocess_speculative(scheduler, seq, [4, 5])

        self.assertEqual(seq.token_ids, [1, 2, 3, 4, 5])
        self.assertEqual(seq.num_cached_tokens, 5)
        self.assertEqual(len(seq.block_table), 2)
        self.assertEqual(len(scheduler.block_manager.free_block_ids), 6)
        self.assertEqual(seq.status, SequenceStatus.RUNNING)

    def test_postprocess_speculative_marks_finished_on_eos(self):
        scheduler = self.make_scheduler()
        seq = Sequence([1, 2, 3], SimpleNamespace(temperature=1.0, max_tokens=8, ignore_eos=False))
        seq.status = SequenceStatus.RUNNING
        seq.num_cached_tokens = len(seq)
        scheduler.running.append(seq)
        scheduler.block_manager.allocate(seq)
        scheduler.block_manager.ensure_slots(seq, num_tokens=3)

        Scheduler.postprocess_speculative(scheduler, seq, [4, 99, 7])

        self.assertEqual(seq.completion_token_ids, [4, 99])
        self.assertEqual(seq.status, SequenceStatus.FINISHED)
        self.assertEqual(scheduler.running, [])
        self.assertEqual(seq.block_table, [])

    def test_postprocess_speculative_respects_max_tokens(self):
        scheduler = self.make_scheduler()
        seq = Sequence([1, 2, 3], SimpleNamespace(temperature=1.0, max_tokens=2, ignore_eos=True))
        seq.status = SequenceStatus.RUNNING
        seq.num_cached_tokens = len(seq)
        scheduler.running.append(seq)
        scheduler.block_manager.allocate(seq)
        scheduler.block_manager.ensure_slots(seq, num_tokens=4)

        Scheduler.postprocess_speculative(scheduler, seq, [4, 5, 6, 7])

        self.assertEqual(seq.completion_token_ids, [4, 5])
        self.assertEqual(seq.status, SequenceStatus.FINISHED)
        self.assertEqual(seq.block_table, [])


if __name__ == "__main__":
    unittest.main()
