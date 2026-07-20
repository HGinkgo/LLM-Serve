import unittest
from collections import deque
from types import SimpleNamespace

from llmserve.engine.block_manager import BlockManager
from llmserve.engine.scheduler import Scheduler, SchedulerOutput
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
        scheduler.max_num_seqs = 4
        scheduler.max_num_batched_tokens = 5
        scheduler.enable_chunked_prefill = True
        scheduler.waiting = deque()
        scheduler.running = deque()
        return scheduler

    def test_schedule_returns_explicit_mixed_batch_groups_without_negative_sentinel(self):
        scheduler = self.make_scheduler()
        running = Sequence([1, 2, 3, 4])
        running.status = SequenceStatus.RUNNING
        running.num_cached_tokens = len(running)
        scheduler.block_manager.allocate(running)
        scheduler.running.append(running)

        waiting = Sequence([5, 6, 7, 8, 9, 10])
        scheduler.waiting.append(waiting)

        output = scheduler.schedule()

        self.assertIsInstance(output, SchedulerOutput)
        self.assertEqual(output.scheduled_seqs, [waiting, running])
        self.assertEqual(output.prefill_seqs, [waiting])
        self.assertEqual(output.decode_seqs, [running])
        self.assertEqual(output.num_batched_tokens, 5)
        self.assertEqual(running.num_scheduled_tokens, 1)
        self.assertEqual(waiting.num_scheduled_tokens, 4)
        self.assertGreaterEqual(running.num_scheduled_tokens, 0)

    def test_scheduler_output_rejects_inconsistent_sequence_order(self):
        seq = Sequence([1])
        with self.assertRaises(ValueError):
            SchedulerOutput(
                scheduled_seqs=[seq],
                prefill_seqs=[],
                decode_seqs=[],
                num_batched_tokens=1,
            )

    def test_postprocess_uses_explicit_prefill_and_decode_groups(self):
        scheduler = self.make_scheduler()
        running = Sequence([1, 2, 3, 4])
        running.status = SequenceStatus.RUNNING
        running.num_cached_tokens = len(running)
        scheduler.block_manager.allocate(running)
        scheduler.running.append(running)
        waiting = Sequence([5, 6, 7, 8, 9, 10])
        scheduler.waiting.append(waiting)

        output = scheduler.schedule()
        scheduler.postprocess(output, [77, 88])

        self.assertEqual(waiting.num_cached_tokens, 4)
        self.assertEqual(waiting.token_ids, [5, 6, 7, 8, 9, 10])
        self.assertEqual(running.completion_token_ids, [88])
        self.assertEqual(running.num_scheduled_tokens, 0)

    def test_non_chunked_prefill_returns_prefill_group(self):
        scheduler = self.make_scheduler()
        scheduler.enable_chunked_prefill = False
        scheduler.max_num_batched_tokens = 8
        seq = Sequence([1, 2, 3])
        scheduler.waiting.append(seq)

        output = scheduler.schedule()

        self.assertEqual(output.prefill_seqs, [seq])
        self.assertEqual(output.decode_seqs, [])
        self.assertEqual(output.num_batched_tokens, 3)

    def test_decode_preempts_a_running_sequence_when_no_block_is_free(self):
        scheduler = self.make_scheduler()
        scheduler.max_num_batched_tokens = 4
        scheduler.block_manager = BlockManager(num_blocks=4, block_size=4)
        seq1 = Sequence([1, 2, 3, 4, 5])
        seq2 = Sequence([6, 7, 8, 9, 10])
        for seq in (seq1, seq2):
            seq.status = SequenceStatus.RUNNING
            seq.num_cached_tokens = len(seq)
            scheduler.block_manager.allocate(seq)
            scheduler.running.append(seq)

        output = scheduler.schedule()

        self.assertEqual(output.prefill_seqs, [seq2])
        self.assertEqual(output.decode_seqs, [seq1])
        self.assertEqual(list(scheduler.waiting), [])
        self.assertEqual(seq1.num_scheduled_tokens, 1)
        self.assertEqual(seq2.num_scheduled_tokens, 1)

    def test_postprocess_uses_explicit_groups_for_mixed_batch(self):
        scheduler = self.make_scheduler()
        running = Sequence([1, 2, 3, 4])
        running.status = SequenceStatus.RUNNING
        running.num_cached_tokens = len(running)
        scheduler.block_manager.allocate(running)
        scheduler.running.append(running)

        waiting = Sequence([5, 6, 7, 8, 9, 10])
        scheduler.waiting.append(waiting)
        output = scheduler.schedule()

        scheduler.postprocess(output, [20, 30])

        self.assertEqual(waiting.num_cached_tokens, 4)
        self.assertEqual(waiting.completion_token_ids, [])
        self.assertEqual(running.completion_token_ids, [30])
        self.assertEqual(running.num_scheduled_tokens, 0)

    def test_non_chunked_schedule_marks_prefill_group(self):
        scheduler = self.make_scheduler()
        scheduler.enable_chunked_prefill = False
        seq = Sequence([1, 2, 3, 4])
        scheduler.waiting.append(seq)

        output = scheduler.schedule()

        self.assertEqual(output.scheduled_seqs, [seq])
        self.assertEqual(output.prefill_seqs, [seq])
        self.assertEqual(output.decode_seqs, [])
        self.assertEqual(output.num_batched_tokens, 4)

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
        self.assertEqual(list(scheduler.running), [])
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
