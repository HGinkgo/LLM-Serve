import unittest

import torch

from llmserve.speculative.tree_kv import (
    TreeKVCacheManager,
    fused_commit_tree_kv,
)


class FakeTreeAttention:

    def __init__(self, values):
        self.tree_k = torch.tensor(values, dtype=torch.float32).view(-1, 1, 1)
        self.tree_v = self.tree_k + 10
        self.k_cache = torch.zeros(1, 8, 1, 1)
        self.v_cache = torch.zeros(1, 8, 1, 1)

class TreeKVCacheManagerTest(unittest.TestCase):

    def test_commit_rejects_mismatched_layer_shapes_and_clears(self):
        layers = [FakeTreeAttention([1, 2, 3]), FakeTreeAttention([4, 5])]
        manager = TreeKVCacheManager(layers)

        with self.assertRaisesRegex(ValueError, "same shape"):
            manager.commit([0], torch.tensor([4], dtype=torch.int32))

        for layer in layers:
            self.assertEqual(layer.tree_k.numel(), 0)
            self.assertEqual(layer.tree_v.numel(), 0)

    def test_commit_rejects_invalid_indices_and_slot_count(self):
        layer = FakeTreeAttention([1, 2, 3])
        manager = TreeKVCacheManager([layer])

        with self.assertRaisesRegex(ValueError, "slot count"):
            manager.commit([0, 2], torch.tensor([4], dtype=torch.int32))

        layer.tree_k = torch.tensor([1, 2, 3], dtype=torch.float32).view(-1, 1, 1)
        layer.tree_v = layer.tree_k + 10
        with self.assertRaisesRegex(ValueError, "node index"):
            manager.commit([0, 3], torch.tensor([4, 5], dtype=torch.int32))

        self.assertEqual(layer.tree_k.numel(), 0)
        self.assertEqual(layer.tree_v.numel(), 0)

    def test_manager_does_not_expose_completed_ablation_mode(self):
        with self.assertRaises(TypeError):
            TreeKVCacheManager([FakeTreeAttention([1])], mode="layerwise")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_manager_fused_commit_writes_selected_path_and_clears(self):
        layers = [FakeTreeAttention([1, 2, 3]), FakeTreeAttention([4, 5, 6])]
        for layer in layers:
            layer.tree_k = layer.tree_k.cuda()
            layer.tree_v = layer.tree_v.cuda()
        k_cache = torch.zeros(2, 1, 8, 1, 1, device="cuda")
        v_cache = torch.zeros_like(k_cache)
        manager = TreeKVCacheManager(layers, k_cache=k_cache, v_cache=v_cache)

        manager.commit([0, 2], torch.tensor([4, 5], dtype=torch.int32, device="cuda"))
        torch.cuda.synchronize()

        self.assertEqual(k_cache[:, 0, 4:6, 0, 0].tolist(), [[1.0, 3.0], [4.0, 6.0]])
        self.assertEqual(v_cache[:, 0, 4:6, 0, 0].tolist(), [[11.0, 13.0], [14.0, 16.0]])
        for layer in layers:
            self.assertEqual(layer.tree_k.numel(), 0)
            self.assertEqual(layer.tree_v.numel(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_fused_commit_matches_reference_for_small_shape(self):
        tree_k = torch.arange(12, device="cuda", dtype=torch.float32).view(2, 3, 1, 2)
        tree_v = tree_k + 100
        k_cache = torch.zeros(2, 2, 4, 1, 2, device="cuda")
        v_cache = torch.zeros_like(k_cache)
        node_indices = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
        slots = torch.tensor([1, 6], dtype=torch.int32, device="cuda")
        expected_k = k_cache.clone().view(2, 8, 1, 2)
        expected_v = v_cache.clone().view(2, 8, 1, 2)
        expected_k[:, slots.long()] = tree_k[:, node_indices.long()]
        expected_v[:, slots.long()] = tree_v[:, node_indices.long()]

        fused_commit_tree_kv(tree_k, tree_v, k_cache, v_cache, node_indices, slots)

        torch.cuda.synchronize()
        self.assertTrue(torch.equal(k_cache.view_as(expected_k), expected_k))
        self.assertTrue(torch.equal(v_cache.view_as(expected_v), expected_v))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_fused_commit_matches_qwen3_kv_shape(self):
        shape = (36, 7, 8, 128)
        tree_k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        tree_v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        k_cache = torch.zeros(36, 2, 4, 8, 128, device="cuda", dtype=torch.bfloat16)
        v_cache = torch.zeros_like(k_cache)
        node_indices = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
        slots = torch.tensor([1, 5, 7], dtype=torch.int32, device="cuda")
        expected_k = k_cache.clone().view(36, 8, 8, 128)
        expected_v = v_cache.clone().view(36, 8, 8, 128)
        expected_k[:, slots.long()] = tree_k[:, node_indices.long()]
        expected_v[:, slots.long()] = tree_v[:, node_indices.long()]

        fused_commit_tree_kv(tree_k, tree_v, k_cache, v_cache, node_indices, slots)

        torch.cuda.synchronize()
        self.assertTrue(torch.equal(k_cache.view_as(expected_k), expected_k))
        self.assertTrue(torch.equal(v_cache.view_as(expected_v), expected_v))


if __name__ == "__main__":
    unittest.main()
