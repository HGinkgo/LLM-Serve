import unittest

import torch

from llmserve.models.eagle3 import Eagle3DraftOutput
from llmserve.speculative.tree import (
    build_fixed_tree_topology,
    generate_eagle3_draft_tree,
    select_greedy_tree_path,
)


class FakeTreeDraftModel:
    hidden_size = 4

    def __init__(self):
        self.calls = []

    def greedy_topk(self, draft_logits, k):
        return torch.topk(draft_logits, k=k, dim=-1).indices

    def propose(self, input_ids, aux_hidden_states, positions, temperature, past_kv=None, kv_valid_lens=None):
        batch_size = input_ids.size(0)
        self.calls.append({
            "input_ids": input_ids.flatten().tolist(),
            "positions": positions.flatten().tolist(),
            "kv_valid_lens": None if kv_valid_lens is None else kv_valid_lens.tolist(),
        })
        draft_logits = torch.full((batch_size, 1, 32), -100.0)
        for index, input_id in enumerate(input_ids.flatten().tolist()):
            draft_logits[index, 0, input_id + 1] = 1.0
            draft_logits[index, 0, input_id + 2] = 0.0
        hidden_states = input_ids.float().view(batch_size, 1, 1).expand(-1, -1, self.hidden_size)
        if past_kv is None:
            past_k = torch.empty(batch_size, 1, 0, 1)
            past_v = torch.empty(batch_size, 1, 0, 1)
        else:
            past_k, past_v = past_kv
        new_value = positions.float().view(batch_size, 1, 1, 1)
        new_past = (
            torch.cat([past_k, new_value], dim=2),
            torch.cat([past_v, new_value + 100], dim=2),
        )
        return Eagle3DraftOutput(
            hidden_states=hidden_states,
            draft_logits=draft_logits,
            target_logits=draft_logits.clone(),
            token_ids=torch.zeros(batch_size, 1, dtype=torch.long),
            past_kv=new_past,
        )


class Eagle3TreeTest(unittest.TestCase):

    def test_tree6_topology_has_expected_parents_depths_and_mask(self):
        topology = build_fixed_tree_topology(6)

        self.assertEqual(topology.parents, (-1, 0, 0, 1, 2, 3, 4))
        self.assertEqual(topology.depths, (0, 1, 1, 2, 2, 3, 3))
        self.assertEqual(topology.num_draft_nodes, 6)
        self.assertEqual(
            topology.attention_mask().tolist(),
            [
                [True, False, False, False, False, False, False],
                [True, True, False, False, False, False, False],
                [True, False, True, False, False, False, False],
                [True, True, False, True, False, False, False],
                [True, False, True, False, True, False, False],
                [True, True, False, True, False, True, False],
                [True, False, True, False, True, False, True],
            ],
        )

    def test_tree10_branches_at_first_two_depths(self):
        topology = build_fixed_tree_topology(10)

        self.assertEqual(topology.parents, (-1, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6))
        self.assertEqual(topology.depths, (0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3))
        self.assertEqual(topology.children(0), (1, 2))
        self.assertEqual(topology.children(1), (3, 4))
        self.assertEqual(topology.children(3), (7,))

    def test_greedy_selection_follows_matching_branch_and_emits_bonus(self):
        topology = build_fixed_tree_topology(6)
        draft_token_ids = [11, 21, 12, 22, 13, 23]
        target_logits = torch.full((7, 32), -100.0)
        target_logits[0, 21] = 0.0
        target_logits[2, 22] = 0.0
        target_logits[4, 23] = 0.0
        target_logits[6, 31] = 0.0

        result = select_greedy_tree_path(topology, draft_token_ids, target_logits)

        self.assertEqual(result.token_ids, [21, 22, 23, 31])
        self.assertEqual(result.accepted_node_indices, [2, 4, 6])
        self.assertEqual(result.commit_node_indices, [0, 2, 4, 6])
        self.assertEqual(result.num_accepted, 3)
        self.assertTrue(result.accepted_all)
        self.assertEqual(result.final_node_index, 6)

    def test_greedy_selection_stops_at_first_missing_child(self):
        topology = build_fixed_tree_topology(6)
        draft_token_ids = [11, 21, 12, 22, 13, 23]
        target_logits = torch.full((7, 32), -100.0)
        target_logits[0, 11] = 0.0
        target_logits[1, 30] = 0.0

        result = select_greedy_tree_path(topology, draft_token_ids, target_logits)

        self.assertEqual(result.token_ids, [11, 30])
        self.assertEqual(result.accepted_node_indices, [1])
        self.assertEqual(result.commit_node_indices, [0, 1])
        self.assertEqual(result.num_accepted, 1)
        self.assertFalse(result.accepted_all)
        self.assertEqual(result.final_node_index, 1)

    def test_tree6_draft_generation_batches_each_depth(self):
        topology = build_fixed_tree_topology(6)
        model = FakeTreeDraftModel()
        initial_past = (
            torch.zeros(1, 1, 2, 1),
            torch.zeros(1, 1, 2, 1),
        )

        tree = generate_eagle3_draft_tree(
            model,
            topology=topology,
            start_token_id=0,
            start_aux_hidden=torch.zeros(1, 1, 12),
            start_position=2,
            temperature=1.0,
            past_kv=initial_past,
        )

        self.assertEqual(tree.draft_token_ids, [1, 2, 2, 3, 3, 4])
        self.assertEqual([len(call["input_ids"]) for call in model.calls], [1, 2, 2])
        self.assertEqual([call["positions"] for call in model.calls], [[2], [3, 3], [4, 4]])
        selected_past = tree.past_kv_for_path([1, 3, 5])
        self.assertEqual(selected_past[0].shape[2], 5)

    def test_tree10_draft_generation_branches_again_at_depth_two(self):
        topology = build_fixed_tree_topology(10)
        model = FakeTreeDraftModel()

        tree = generate_eagle3_draft_tree(
            model,
            topology=topology,
            start_token_id=0,
            start_aux_hidden=torch.zeros(1, 1, 12),
            start_position=0,
            temperature=1.0,
        )

        self.assertEqual(tree.draft_token_ids, [1, 2, 2, 3, 3, 4, 3, 4, 4, 5])
        self.assertEqual([len(call["input_ids"]) for call in model.calls], [1, 2, 4])
        self.assertEqual(tree.past_kv_for_path([2, 6, 10])[0].shape[2], 3)


if __name__ == "__main__":
    unittest.main()
