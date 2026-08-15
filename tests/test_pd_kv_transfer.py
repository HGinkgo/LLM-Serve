import unittest

import torch

from llmserve.pd.kv_transfer import (
    export_logical_kv,
    import_logical_kv,
    summarize_cuda_overlap,
)


class TestPDKVTransfer(unittest.TestCase):

    class FakeEvent:

        def __init__(self, timestamp_ms):
            self.timestamp_ms = timestamp_ms

        def elapsed_time(self, other):
            return other.timestamp_ms - self.timestamp_ms

    class FakeInterval:

        def __init__(self, start_ms, end_ms, device="cuda:0"):
            self.device = device
            self.start_event = TestPDKVTransfer.FakeEvent(start_ms)
            self.end_event = TestPDKVTransfer.FakeEvent(end_ms)

        def is_complete(self):
            return True

        def elapsed_ms(self):
            return self.start_event.elapsed_time(self.end_event)

    @staticmethod
    def make_cache():
        cache = torch.zeros(2, 2, 3, 4, 1, 2, dtype=torch.float32)
        for kv_index in range(cache.size(0)):
            for layer_index in range(cache.size(1)):
                for block_index in range(cache.size(2)):
                    for offset in range(cache.size(3)):
                        cache[kv_index, layer_index, block_index, offset].fill_(
                            kv_index * 1000
                            + layer_index * 100
                            + block_index * 10
                            + offset
                        )
        return cache

    def test_export_reads_logical_tokens_across_physical_blocks(self):
        cache = self.make_cache()

        payload = export_logical_kv(cache, [2, 0], num_tokens=6)

        self.assertEqual(payload.shape, (2, 2, 6, 1, 2))
        self.assertEqual(payload[0, 1, :, 0, 0].tolist(), [120, 121, 122, 123, 100, 101])

    def test_import_writes_logical_tokens_into_target_blocks(self):
        source = self.make_cache()
        payload = export_logical_kv(source, [2, 0], num_tokens=6)
        target = torch.zeros_like(source)

        import_logical_kv(target, [1, 2], payload)

        restored = export_logical_kv(target, [1, 2], num_tokens=6)
        self.assertTrue(torch.equal(restored, payload))

    def test_rejects_payload_or_block_table_that_cannot_cover_tokens(self):
        cache = self.make_cache()
        with self.assertRaises(ValueError):
            export_logical_kv(cache, [0], num_tokens=5)
        with self.assertRaises(ValueError):
            import_logical_kv(cache, [0], torch.zeros(2, 2, 5, 1, 2))

    def test_cuda_overlap_reports_device_timeline_intersection(self):
        transfer = self.FakeInterval(0.0, 1.0)
        compute = self.FakeInterval(0.2, 4.2)

        metrics = summarize_cuda_overlap(transfer, compute)

        self.assertEqual(metrics["copy_gpu_ms"], 1.0)
        self.assertEqual(metrics["decode_gpu_step_ms"], 4.0)
        self.assertNotIn("decode_compute_gpu_ms", metrics)
        self.assertAlmostEqual(metrics["copy_compute_overlap_ms"], 0.8)
        self.assertAlmostEqual(metrics["copy_compute_overlap_ratio"], 0.8)
        self.assertAlmostEqual(metrics["serial_gpu_ms"], 5.0)
        self.assertAlmostEqual(metrics["overlapped_makespan_gpu_ms"], 4.2)
        self.assertAlmostEqual(metrics["critical_path_reduction_gpu_ms"], 0.8)


if __name__ == "__main__":
    unittest.main()
