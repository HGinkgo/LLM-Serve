import unittest

from llmserve.engine.block_manager import BlockManager
from llmserve.engine.sequence import Sequence


class BlockManagerSpeculativeTest(unittest.TestCase):

    def setUp(self):
        self.old_block_size = Sequence.block_size
        Sequence.block_size = 4

    def tearDown(self):
        Sequence.block_size = self.old_block_size

    def test_ensure_slots_allocates_future_blocks_for_multiple_tokens(self):
        manager = BlockManager(num_blocks=4, block_size=4)
        seq = Sequence([1, 2, 3])
        manager.allocate(seq)

        self.assertTrue(manager.ensure_slots(seq, num_tokens=5))

        self.assertEqual(len(seq.block_table), 2)
        self.assertEqual(len(manager.free_block_ids), 2)

    def test_ensure_slots_batch_is_atomic_when_total_capacity_is_insufficient(self):
        manager = BlockManager(num_blocks=3, block_size=4)
        seq1 = Sequence([1, 2, 3])
        seq2 = Sequence([4, 5, 6])
        manager.allocate(seq1)
        manager.allocate(seq2)
        original_tables = [list(seq1.block_table), list(seq2.block_table)]

        self.assertFalse(manager.ensure_slots_batch([seq1, seq2], num_tokens=5))

        self.assertEqual(seq1.block_table, original_tables[0])
        self.assertEqual(seq2.block_table, original_tables[1])
        self.assertEqual(len(manager.free_block_ids), 1)

    def test_ensure_slots_batch_reserves_each_sequence(self):
        manager = BlockManager(num_blocks=6, block_size=4)
        seq1 = Sequence([1, 2, 3])
        seq2 = Sequence([4, 5, 6])
        manager.allocate(seq1)
        manager.allocate(seq2)

        self.assertTrue(manager.ensure_slots_batch([seq1, seq2], num_tokens=5))

        self.assertEqual(len(seq1.block_table), 2)
        self.assertEqual(len(seq2.block_table), 2)
        self.assertEqual(len(manager.free_block_ids), 2)

    def test_finalize_full_blocks_hashes_blocks_after_multi_token_append(self):
        manager = BlockManager(num_blocks=4, block_size=4)
        seq = Sequence([1, 2, 3])
        manager.allocate(seq)
        manager.ensure_slots(seq, num_tokens=5)

        for token_id in [4, 5, 6, 7, 8]:
            seq.append_token(token_id)
        manager.finalize_full_blocks(seq)

        block0 = manager.blocks[seq.block_table[0]]
        block1 = manager.blocks[seq.block_table[1]]
        self.assertNotEqual(block0.hash, -1)
        self.assertNotEqual(block1.hash, -1)
        self.assertEqual(block0.token_ids, [1, 2, 3, 4])
        self.assertEqual(block1.token_ids, [5, 6, 7, 8])

    def test_may_append_is_idempotent_after_speculative_block_finalization(self):
        manager = BlockManager(num_blocks=4, block_size=4)
        seq = Sequence([1, 2, 3])
        manager.allocate(seq)
        seq.append_token(4)
        manager.finalize_full_blocks(seq)
        original_table = list(seq.block_table)
        original_hash = manager.blocks[seq.block_table[-1]].hash

        manager.may_append(seq)

        self.assertEqual(seq.block_table, original_table)
        self.assertEqual(manager.blocks[seq.block_table[-1]].hash, original_hash)
        self.assertEqual(len(manager.free_block_ids), 3)

        seq.append_token(5)
        manager.may_append(seq)
        self.assertEqual(len(seq.block_table), 2)
        self.assertEqual(len(manager.free_block_ids), 2)

    def test_may_append_reuses_speculatively_reserved_next_block(self):
        manager = BlockManager(num_blocks=4, block_size=4)
        seq = Sequence([1, 2, 3])
        manager.allocate(seq)
        manager.ensure_slots(seq, num_tokens=2)
        seq.append_token(4)
        seq.append_token(5)
        manager.finalize_full_blocks(seq)
        manager.release_extra_slots(seq)
        original_table = list(seq.block_table)

        manager.may_append(seq)

        self.assertEqual(seq.block_table, original_table)
        self.assertEqual(len(manager.free_block_ids), 2)

    def test_release_extra_slots_removes_unused_reserved_blocks(self):
        manager = BlockManager(num_blocks=4, block_size=4)
        seq = Sequence([1, 2, 3])
        manager.allocate(seq)
        manager.ensure_slots(seq, num_tokens=5)

        seq.append_token(4)
        manager.finalize_full_blocks(seq)
        manager.release_extra_slots(seq)

        self.assertEqual(len(seq.block_table), 1)
        self.assertEqual(len(manager.free_block_ids), 3)


if __name__ == "__main__":
    unittest.main()
