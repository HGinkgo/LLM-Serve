import unittest

import torch

from llmserve.pd.kv_transfer import (
    export_logical_kv,
    import_logical_kv,
)


class TestPDKVTransfer(unittest.TestCase):

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


if __name__ == "__main__":
    unittest.main()
