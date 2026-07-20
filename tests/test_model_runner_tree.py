import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from llmserve.engine.model_runner import ModelRunner
from llmserve.engine.sequence import Sequence
from llmserve.models.eagle3 import Eagle3TargetVerifyOutput
from llmserve.speculative.tree import Eagle3DraftTree, build_fixed_tree_topology


class FakeTreeKVManager:

    def __init__(self):
        self.calls = []
        self.device = torch.device("cpu")

    def commit(self, node_indices, slot_mapping):
        self.calls.append((list(node_indices), slot_mapping.tolist()))


class ModelRunnerTreeTest(unittest.TestCase):

    def setUp(self):
        self.old_block_size = Sequence.block_size
        Sequence.block_size = 4

    def tearDown(self):
        Sequence.block_size = self.old_block_size

    def test_build_tree_verify_metadata_uses_depth_positions_and_paged_prefix(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.block_size = 4
        seq = Sequence([1, 2, 3, 4, 5])
        seq.block_table = [10, 20, 30]
        topology = build_fixed_tree_topology(6)

        metadata = ModelRunner._build_target_tree_verify_metadata(
            runner,
            seq,
            start_token_id=99,
            draft_token_ids=[11, 21, 12, 22, 13, 23],
            topology=topology,
            base_offset=0,
        )

        self.assertEqual(metadata["input_ids"], [99, 11, 21, 12, 22, 13, 23])
        self.assertEqual(metadata["positions"], [5, 6, 6, 7, 7, 8, 8])
        self.assertEqual(metadata["prefix_slots"], [40, 41, 42, 43, 80])
        self.assertEqual(metadata["base_pos"], 5)
        self.assertEqual(metadata["attention_mask"].shape, (7, 7))

    def test_commit_target_tree_kv_writes_only_root_and_accepted_path(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.block_size = 4
        manager = FakeTreeKVManager()
        runner.tree_kv_cache_manager = manager
        seq = Sequence([1, 2, 3, 4, 5])
        seq.block_table = [10, 20, 30]

        ModelRunner._commit_target_tree_kv(
            runner,
            seq,
            base_pos=5,
            node_indices=[0, 2, 4, 6],
        )

        self.assertEqual(manager.calls, [([0, 2, 4, 6], [81, 82, 83, 120])])

    def test_run_speculative_tree_single_selects_and_commits_matching_branch(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.draft_model = object()
        runner.speculative_gamma = 3
        runner.speculative_tree_nodes = 6
        runner.speculative_accept_mode = "greedy"
        runner.speculative_trace = False
        runner.draft_kv_cache = {
            0: (torch.zeros(1, 1, 5, 1), torch.zeros(1, 1, 5, 1)),
        }
        runner._prev_correction = {}
        runner.run_target_decode_with_eagle3_aux = lambda seqs: SimpleNamespace(
            token_ids=[10],
            aux_hidden=torch.zeros(1, 12),
            positions=torch.tensor([4]),
        )
        topology = build_fixed_tree_topology(6)
        processed_past = [None] * 7
        processed_past[0] = (torch.zeros(1, 1, 6, 1), torch.zeros(1, 1, 6, 1))
        processed_past[2] = (torch.zeros(1, 1, 7, 1), torch.zeros(1, 1, 7, 1))
        processed_past[4] = (torch.zeros(1, 1, 8, 1), torch.zeros(1, 1, 8, 1))
        draft_tree = Eagle3DraftTree(
            topology=topology,
            draft_token_ids=[11, 21, 12, 22, 13, 23],
            processed_past_kv=processed_past,
        )
        target_logits = torch.full((7, 32), -100.0)
        target_logits[0, 21] = 0.0
        target_logits[2, 22] = 0.0
        target_logits[4, 23] = 0.0
        target_logits[6, 31] = 0.0
        target_aux = torch.arange(7, dtype=torch.float32).view(7, 1).expand(-1, 12)
        runner.run_target_verify_tree_with_eagle3_aux = lambda *args, **kwargs: (
            Eagle3TargetVerifyOutput(target_logits, target_aux),
            5,
        )
        committed = []
        runner._commit_target_tree_kv = lambda seq, base_pos, node_indices: committed.append(
            (base_pos, list(node_indices))
        )
        seq = Sequence([1, 2, 3, 4, 5])
        seq.seq_id = 0

        with patch(
            "llmserve.engine.speculative_executor.generate_eagle3_draft_tree",
            return_value=draft_tree,
        ):
            output = ModelRunner.run_speculative_tree_single(runner, seq)

        self.assertEqual(output.token_ids, [10, 21, 22, 23, 31])
        self.assertEqual(output.num_draft_tokens, 6)
        self.assertEqual(output.num_accepted, 3)
        self.assertTrue(output.accepted_all)
        self.assertEqual(committed, [(5, [0, 2, 4, 6])])
        self.assertEqual(runner.draft_kv_cache[0][0].shape[2], 8)
        self.assertEqual(runner._prev_correction[0][0], 31)
        self.assertEqual(float(runner._prev_correction[0][1][0]), 6.0)
        self.assertIn("target_tree_kv_commit_time", output.timing)
        self.assertIn("draft_kv_update_time", output.timing)
        self.assertEqual(
            output.timing["kv_update_time"],
            output.timing["target_tree_kv_commit_time"]
            + output.timing["draft_kv_update_time"],
        )


if __name__ == "__main__":
    unittest.main()
