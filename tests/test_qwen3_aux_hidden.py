import unittest

import torch
from torch import nn

from llmserve.models.qwen3 import Qwen3ForCausalLM, Qwen3Model, get_eagle3_aux_layer_ids


class FakeEmbedding(nn.Module):

    def forward(self, input_ids):
        return torch.zeros(input_ids.numel(), 4)


class FakeLayer(nn.Module):

    def __init__(self, layer_id: int):
        super().__init__()
        self.layer_id = layer_id

    def forward(self, positions, hidden_states, residual):
        hidden_states = torch.full_like(hidden_states, float(self.layer_id))
        residual = torch.full_like(hidden_states, float(100 + self.layer_id))
        return hidden_states, residual


class FakeNorm(nn.Module):

    def forward(self, hidden_states, residual):
        return hidden_states + residual, residual


class Qwen3AuxHiddenTest(unittest.TestCase):

    def test_default_eagle3_aux_layer_ids_for_qwen3_8b(self):
        self.assertEqual(get_eagle3_aux_layer_ids(36), (2, 18, 33))

    def test_forward_can_return_three_aux_layers_before_selected_layers_without_changing_normal_path(self):
        model = Qwen3Model.__new__(Qwen3Model)
        nn.Module.__init__(model)
        model.embed_tokens = FakeEmbedding()
        model.layers = nn.ModuleList(FakeLayer(i) for i in range(8))
        model.norm = FakeNorm()

        input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        positions = torch.tensor([0, 1, 2], dtype=torch.long)

        normal_hidden = Qwen3Model.forward(model, input_ids, positions)
        hidden, aux_hidden = Qwen3Model.forward(model, input_ids, positions, aux_layer_ids={2, 4, 5})

        self.assertIsInstance(normal_hidden, torch.Tensor)
        self.assertTrue(torch.equal(hidden, normal_hidden))
        self.assertEqual(aux_hidden.shape, (3, 12))

        expected = torch.cat([
            torch.full((3, 4), 102.0),
            torch.full((3, 4), 106.0),
            torch.full((3, 4), 108.0),
        ], dim=-1)
        self.assertTrue(torch.equal(aux_hidden, expected))

    def test_causal_lm_forward_with_eagle3_aux_uses_default_layers(self):
        model = Qwen3Model.__new__(Qwen3Model)
        nn.Module.__init__(model)
        model.embed_tokens = FakeEmbedding()
        model.layers = nn.ModuleList(FakeLayer(i) for i in range(8))
        model.norm = FakeNorm()

        causal_lm = Qwen3ForCausalLM.__new__(Qwen3ForCausalLM)
        nn.Module.__init__(causal_lm)
        causal_lm.model = model

        input_ids = torch.tensor([1], dtype=torch.long)
        positions = torch.tensor([0], dtype=torch.long)

        hidden, aux_hidden = Qwen3ForCausalLM.forward_with_eagle3_aux(causal_lm, input_ids, positions)

        self.assertEqual(hidden.shape, (1, 4))
        self.assertEqual(aux_hidden.shape, (1, 12))
        expected = torch.cat([
            torch.full((1, 4), 102.0),
            torch.full((1, 4), 106.0),
            torch.full((1, 4), 108.0),
        ], dim=-1)
        self.assertTrue(torch.equal(aux_hidden, expected))

    def test_compute_logits_can_return_all_token_logits_for_verify(self):
        causal_lm = Qwen3ForCausalLM.__new__(Qwen3ForCausalLM)
        nn.Module.__init__(causal_lm)
        causal_lm.lm_head = nn.Linear(4, 3, bias=False)
        causal_lm.lm_head.weight.data.copy_(torch.eye(3, 4))

        hidden_states = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ])

        logits = Qwen3ForCausalLM.compute_logits(causal_lm, hidden_states, all_tokens=True)

        self.assertEqual(logits.tolist(), [[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]])


if __name__ == "__main__":
    unittest.main()
