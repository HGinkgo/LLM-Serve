import unittest

import torch

from llmserve.speculative.target_graph import (
    TargetVerifyGraphKey,
    TargetVerifyGraphPlan,
    TargetVerifyGraphWorkspace,
)


class TargetVerifyGraphPlanTest(unittest.TestCase):

    def setUp(self):
        self.plan = TargetVerifyGraphPlan(
            gamma=3,
            max_batch_size=8,
            max_model_len=4096,
        )

    def test_selects_smallest_batch_and_context_buckets(self):
        key = self.plan.select(
            verify_lengths=[4, 4, 4, 4],
            max_seqlen_k=700,
        )

        self.assertEqual(key, TargetVerifyGraphKey(batch_size=4, context_frontier=1024))

    def test_rejects_variable_verify_lengths(self):
        self.assertIsNone(
            self.plan.select(verify_lengths=[4, 3], max_seqlen_k=300)
        )

    def test_rejects_batch_or_context_outside_captured_range(self):
        self.assertIsNone(
            self.plan.select(verify_lengths=[4] * 9, max_seqlen_k=300)
        )
        self.assertIsNone(
            self.plan.select(verify_lengths=[4], max_seqlen_k=4097)
        )

    def test_large_runtime_batch_limit_does_not_add_a_huge_graph(self):
        plan = TargetVerifyGraphPlan(
            gamma=3,
            max_batch_size=512,
            max_model_len=1024,
        )
        self.assertEqual(plan.batch_buckets, (1, 4, 8))
        self.assertIsNone(plan.select([4] * 512, max_seqlen_k=300))

    def test_includes_nonstandard_model_limit_as_last_frontier(self):
        plan = TargetVerifyGraphPlan(
            gamma=3,
            max_batch_size=4,
            max_model_len=1536,
        )

        self.assertEqual(plan.context_frontiers, (256, 1024))
        self.assertEqual(
            plan.select([4], 1400),
            None,
        )

    def test_caps_capture_frontier_to_keep_graph_memory_bounded(self):
        plan = TargetVerifyGraphPlan(
            gamma=3,
            max_batch_size=8,
            max_model_len=4096,
        )

        self.assertEqual(plan.context_frontiers, (256, 1024))
        self.assertIsNone(plan.select([4], 1025))


class TargetVerifyGraphWorkspaceTest(unittest.TestCase):

    def test_loads_metadata_into_fixed_buffers(self):
        workspace = TargetVerifyGraphWorkspace(
            TargetVerifyGraphKey(batch_size=2, context_frontier=512),
            verify_width=4,
            block_size=256,
            device="cpu",
        )
        metadata = {
            "input_ids": list(range(8)),
            "positions": list(range(100, 108)),
            "slot_mapping": list(range(200, 208)),
            "cu_seqlens_q": [0, 4, 8],
            "cu_seqlens_k": [0, 104, 212],
        }
        block_tables = torch.tensor([[7, -1], [9, 10]], dtype=torch.int32)

        workspace.load(metadata, block_tables)

        self.assertEqual(workspace.input_ids.tolist(), metadata["input_ids"])
        self.assertEqual(workspace.positions.tolist(), metadata["positions"])
        self.assertEqual(workspace.slot_mapping.tolist(), metadata["slot_mapping"])
        self.assertEqual(workspace.cu_seqlens_q.tolist(), [0, 4, 8])
        self.assertEqual(workspace.cu_seqlens_k.tolist(), [0, 104, 212])
        self.assertEqual(workspace.block_tables.tolist(), [[7, -1], [9, 10]])

    def test_rejects_metadata_that_does_not_fit_graph_key(self):
        workspace = TargetVerifyGraphWorkspace(
            TargetVerifyGraphKey(batch_size=1, context_frontier=256),
            verify_width=4,
            block_size=256,
            device="cpu",
        )
        metadata = {
            "input_ids": [1, 2, 3],
            "positions": [0, 1, 2],
            "slot_mapping": [0, 1, 2],
            "cu_seqlens_q": [0, 3],
            "cu_seqlens_k": [0, 3],
        }

        with self.assertRaisesRegex(ValueError, "token count"):
            workspace.load(
                metadata,
                torch.tensor([[0]], dtype=torch.int32),
            )

    def test_reserved_block_overflow_is_eager_fallback(self):
        from llmserve.speculative.target_graph import TargetVerifyGraphBackend

        backend = TargetVerifyGraphBackend.__new__(TargetVerifyGraphBackend)
        backend.plan = TargetVerifyGraphPlan(
            gamma=3,
            max_batch_size=8,
            max_model_len=1024,
        )
        backend.graphs = {TargetVerifyGraphKey(1, 1024): object()}
        backend.workspaces = {
            TargetVerifyGraphKey(1, 1024): TargetVerifyGraphWorkspace(
                TargetVerifyGraphKey(1, 1024),
                verify_width=4,
                block_size=256,
                device="cpu",
            )
        }
        backend.graph_replays = 0
        backend.eager_fallbacks = 0
        backend.graph_replays_by_key = {}
        backend.fallback_reasons = {}
        backend.last_key = None

        result = backend.run_if_supported(
            {
                "verify_lengths": [4],
                "max_seqlen_k": 1024,
                "input_ids": [1, 2, 3, 4],
                "positions": [1020, 1021, 1022, 1023],
                "slot_mapping": [0, 1, 2, 3],
                "cu_seqlens_q": [0, 4],
                "cu_seqlens_k": [0, 1024],
            },
            torch.zeros((1, 5), dtype=torch.int32),
        )

        self.assertIsNone(result)
        self.assertEqual(backend.fallback_reasons, {"block_table_capacity": 1})


if __name__ == "__main__":
    unittest.main()
