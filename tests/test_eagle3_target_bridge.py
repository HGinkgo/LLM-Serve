import json
import os
import unittest

import torch
from torch import nn

from llmserve.models.eagle3 import Eagle3Speculator
from llmserve.models.qwen3 import Qwen3ForCausalLM, Qwen3Model
from tests.support import (
    SPECULATIVE_MODEL_PATH,
    TARGET_MODEL_PATH,
    requires_eagle3_models,
)


class FakeQwen3Embedding(nn.Module):

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, input_ids):
        return torch.zeros(input_ids.numel(), self.hidden_size)


class FakeQwen3Layer(nn.Module):

    def __init__(self, layer_id: int):
        super().__init__()
        self.layer_id = layer_id

    def forward(self, positions, hidden_states, residual):
        hidden_states = torch.full_like(hidden_states, float(self.layer_id))
        residual = torch.full_like(hidden_states, float(100 + self.layer_id))
        return hidden_states, residual


class FakeQwen3Norm(nn.Module):

    def forward(self, hidden_states, residual):
        return hidden_states + residual, residual


class Eagle3TargetBridgeTest(unittest.TestCase):

    @requires_eagle3_models
    def test_qwen3_aux_hidden_can_drive_first_eagle3_proposal(self):
        draft = Eagle3Speculator.from_pretrained(
            SPECULATIVE_MODEL_PATH,
            target_model_path=TARGET_MODEL_PATH,
        ).float().eval()
        with open(os.path.join(TARGET_MODEL_PATH, "config.json")) as f:
            target_config = json.load(f)

        qwen = Qwen3Model.__new__(Qwen3Model)
        nn.Module.__init__(qwen)
        qwen.embed_tokens = FakeQwen3Embedding(target_config["hidden_size"])
        qwen.layers = nn.ModuleList(
            FakeQwen3Layer(i) for i in range(target_config["num_hidden_layers"])
        )
        qwen.norm = FakeQwen3Norm()

        target = Qwen3ForCausalLM.__new__(Qwen3ForCausalLM)
        nn.Module.__init__(target)
        target.model = qwen

        input_ids = torch.tensor([151644], dtype=torch.long)
        positions = torch.tensor([0], dtype=torch.long)
        _, aux_hidden = Qwen3ForCausalLM.forward_with_eagle3_aux(target, input_ids, positions)

        torch.manual_seed(0)
        with torch.inference_mode():
            output = draft.propose(
                input_ids.view(1, 1),
                aux_hidden.view(1, 1, -1),
                positions.view(1, 1),
                temperature=1.0,
            )

        self.assertEqual(aux_hidden.shape, (1, 3 * target_config["hidden_size"]))
        self.assertEqual(output.token_ids.shape, (1, 1))
        self.assertTrue(bool(draft.t2d[int(output.token_ids.item())]))


if __name__ == "__main__":
    unittest.main()
