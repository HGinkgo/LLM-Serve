import unittest

import torch

from llmserve.pd.protocol import KVTransferDescriptor
from llmserve.pd.shared_slots import (
    KVSlotPoolExhausted,
    KVSlotState,
    SharedKVSlotPool,
    SharedKVSlotReader,
)


class TestSharedKVSlotPool(unittest.TestCase):

    def make_pool(self, slot_count=2, capacity_tokens=8):
        return SharedKVSlotPool.create(
            slot_count=slot_count,
            capacity_tokens=capacity_tokens,
            num_layers=2,
            num_kv_heads=1,
            head_dim=2,
            dtype=torch.float32,
            register_cuda=False,
        )

    def test_slot_becomes_reusable_only_after_every_transfer_is_acked(self):
        pool = self.make_pool(slot_count=1)
        lease = pool.acquire(5)
        pool.mark_ready(lease, {"request-7", "request-8"})
        pool.mark_consuming({"request-7", "request-8"})

        pool.ack("request-7")

        self.assertEqual(pool.slot_state(lease.slot_id), KVSlotState.CONSUMING)
        with self.assertRaises(KVSlotPoolExhausted):
            pool.acquire(1)

        pool.ack("request-8")
        reused = pool.acquire(1)

        self.assertEqual(reused.slot_id, lease.slot_id)
        self.assertEqual(reused.generation, lease.generation + 1)

    def test_reader_exposes_only_the_descriptor_slice(self):
        pool = self.make_pool()
        lease = pool.acquire(5)
        payload = pool.writable_view(lease, token_offset=1, num_tokens=3)
        payload.fill_(7.0)
        pool.mark_ready(lease, {"request-7"})
        reader = SharedKVSlotReader(pool.handle, register_cuda=False)

        actual = reader.read(
            slot_id=lease.slot_id,
            generation=lease.generation,
            token_offset=1,
            num_tokens=3,
        )

        self.assertEqual(tuple(actual.shape), (2, 2, 3, 1, 2))
        self.assertTrue(torch.equal(actual, payload))
        self.assertTrue(payload.is_contiguous())
        self.assertTrue(actual.is_contiguous())

    def test_adjacent_request_slices_are_contiguous_and_do_not_overlap(self):
        pool = self.make_pool()
        lease = pool.acquire(8)

        first = pool.writable_view(lease, token_offset=0, num_tokens=3)
        second = pool.writable_view(lease, token_offset=3, num_tokens=5)
        first.fill_(1.0)
        second.fill_(2.0)

        self.assertTrue(first.is_contiguous())
        self.assertTrue(second.is_contiguous())
        self.assertEqual(float(first.flatten()[-1]), 1.0)
        self.assertEqual(float(second.flatten()[0]), 2.0)

    def test_rejects_capacity_overflow_and_duplicate_ack(self):
        pool = self.make_pool(capacity_tokens=4)

        with self.assertRaisesRegex(ValueError, "capacity"):
            pool.acquire(5)

        lease = pool.acquire(4)
        pool.mark_ready(lease, {"request-7"})
        pool.mark_consuming({"request-7"})
        pool.ack("request-7")

        with self.assertRaisesRegex(ValueError, "unknown transfer"):
            pool.ack("request-7")

    def test_ack_rejects_a_transfer_that_decode_has_not_started_consuming(self):
        pool = self.make_pool()
        lease = pool.acquire(4)
        pool.mark_ready(lease, {"request-7"})

        with self.assertRaisesRegex(ValueError, "not consuming"):
            pool.ack("request-7")

    def test_cancel_releases_a_filling_slot_after_failure(self):
        pool = self.make_pool(slot_count=1)
        lease = pool.acquire(4)

        pool.cancel(lease)
        replacement = pool.acquire(4)

        self.assertEqual(replacement.slot_id, lease.slot_id)
        self.assertEqual(pool.slot_state(replacement.slot_id), KVSlotState.FILLING)

    def test_reader_rejects_filling_and_stale_generations(self):
        pool = self.make_pool(slot_count=1)
        reader = SharedKVSlotReader(pool.handle, register_cuda=False)
        first = pool.acquire(4)

        with self.assertRaisesRegex(ValueError, "not ready"):
            reader.read(
                slot_id=first.slot_id,
                generation=first.generation,
                token_offset=0,
                num_tokens=4,
            )

        pool.mark_ready(first, {"request-7"})
        reader.read(
            slot_id=first.slot_id,
            generation=first.generation,
            token_offset=0,
            num_tokens=4,
        )
        pool.mark_consuming({"request-7"})
        pool.ack("request-7")
        second = pool.acquire(4)
        pool.mark_ready(second, {"request-8"})

        with self.assertRaisesRegex(ValueError, "stale generation"):
            reader.read(
                slot_id=first.slot_id,
                generation=first.generation,
                token_offset=0,
                num_tokens=4,
            )

    def test_reader_rejects_descriptor_geometry_that_differs_from_pool(self):
        pool = self.make_pool()
        lease = pool.acquire(4)
        descriptor = KVTransferDescriptor(
            request_id=7,
            transfer_id="request-7",
            num_tokens=4,
            num_layers=3,
            num_kv_heads=1,
            head_dim=2,
            dtype="float32",
            block_size=4,
            payload_nbytes=192,
            transport="shared_slot",
            slot_id=lease.slot_id,
            slot_generation=lease.generation,
        )
        pool.mark_ready(lease, {descriptor.transfer_id})
        reader = SharedKVSlotReader(pool.handle, register_cuda=False)

        with self.assertRaisesRegex(ValueError, "geometry"):
            reader.read_descriptor(descriptor)


if __name__ == "__main__":
    unittest.main()
