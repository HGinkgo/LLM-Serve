import unittest
import torch

from llmserve.layers.attention import Attention
from llmserve.speculative.tree import Eagle3TreeTopology
from llmserve.utils.context import reset_context, set_context


class TreeAttentionTest(unittest.TestCase):

    def tearDown(self):
        reset_context()

    def test_tree_attention_sees_prefix_and_ancestors_not_siblings(self):
        attention = Attention(num_heads=1, head_dim=1, scale=1.0, num_kv_heads=1)
        attention.k_cache = torch.zeros(1, 4, 1, 1)
        attention.v_cache = torch.zeros(1, 4, 1, 1)
        attention.v_cache[0, 0, 0, 0] = 10.0
        topology = Eagle3TreeTopology((-1, 0, 0))
        set_context(
            True,
            tree_prefix_slots=torch.tensor([0], dtype=torch.long),
            tree_attention_mask=topology.attention_mask(),
        )
        q = torch.zeros(3, 1, 1)
        k = torch.zeros(3, 1, 1)
        v = torch.tensor([[[20.0]], [[30.0]], [[40.0]]])

        output = attention(q, k, v)

        self.assertTrue(torch.allclose(output[:, 0, 0], torch.tensor([15.0, 20.0, 70 / 3])))
        self.assertTrue(torch.equal(attention.tree_k, k))
        self.assertTrue(torch.equal(attention.tree_v, v))
        self.assertEqual(float(attention.v_cache[0, 1, 0, 0]), 0.0)

if __name__ == "__main__":
    unittest.main()
