import json
import os
import tempfile
import unittest

import torch
from safetensors.torch import save_file

import llmserve.models.eagle3 as eagle3
from llmserve.models.eagle3 import Eagle3Speculator
from tests.support import (
    SPECULATIVE_MODEL_PATH,
    TARGET_MODEL_PATH,
    requires_eagle3_models,
)


class Eagle3SpeculatorTest(unittest.TestCase):

    @staticmethod
    def tiny_flat_config():
        return {
            "draft_vocab_size": 4,
            "vocab_size": 8,
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 32,
            "rope_theta": 10000,
            "rms_norm_eps": 1e-6,
        }

    def test_normalizes_nested_redhat_config(self):
        nested = {
            "draft_vocab_size": 32000,
            "target_hidden_size": 4096,
            "norm_before_residual": True,
            "transformer_layer_config": {
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 151936,
            },
        }

        normalized = eagle3.normalize_eagle3_config(nested)

        self.assertEqual(normalized["hidden_size"], 4096)
        self.assertEqual(normalized["draft_vocab_size"], 32000)
        self.assertEqual(normalized["target_hidden_size"], 4096)
        self.assertTrue(normalized["norm_before_residual"])

    def test_normalizes_flat_specforge_config_without_changes(self):
        flat = self.tiny_flat_config()

        self.assertEqual(eagle3.normalize_eagle3_config(flat), flat)

    def test_rejects_conflicting_nested_dimensions(self):
        nested = {
            "hidden_size": 1024,
            "transformer_layer_config": {
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 151936,
            },
        }

        with self.assertRaisesRegex(ValueError, "conflicting EAGLE3 config field: hidden_size"):
            eagle3.normalize_eagle3_config(nested)

    def test_normalizes_redhat_weight_names(self):
        self.assertEqual(
            eagle3.normalize_eagle3_weight_name("layers.0.self_attn.q_proj.weight"),
            "midlayer.self_attn.q_proj.weight",
        )
        self.assertEqual(
            eagle3.normalize_eagle3_weight_name("midlayer.mlp.up_proj.weight"),
            "midlayer.mlp.up_proj.weight",
        )

    def test_rejects_mismatched_target_config(self):
        draft = self.tiny_flat_config()
        target = {
            "hidden_size": 8,
            "vocab_size": draft["vocab_size"],
            "num_hidden_layers": 36,
        }

        with self.assertRaisesRegex(ValueError, "target hidden_size 8 does not match draft target_hidden_size 4"):
            eagle3.validate_eagle3_target_config(draft, target)

    def test_accepts_matching_target_config(self):
        draft = self.tiny_flat_config()
        target = {
            "hidden_size": draft["hidden_size"],
            "vocab_size": draft["vocab_size"],
            "num_hidden_layers": 36,
        }

        eagle3.validate_eagle3_target_config(draft, target)

    def test_loads_nested_redhat_checkpoint_with_layers_prefix(self):
        flat = self.tiny_flat_config()
        source = Eagle3Speculator(flat)
        nested = {
            "draft_vocab_size": flat["draft_vocab_size"],
            "target_hidden_size": flat["hidden_size"],
            "norm_before_residual": True,
            "transformer_layer_config": {
                key: value
                for key, value in flat.items()
                if key != "draft_vocab_size"
            },
        }
        state = {}
        for name, tensor in source.state_dict().items():
            redhat_name = name.replace("midlayer.", "layers.0.")
            state[redhat_name] = tensor.detach().clone().contiguous()

        with tempfile.TemporaryDirectory() as path:
            with open(os.path.join(path, "config.json"), "w") as f:
                json.dump(nested, f)
            save_file(state, os.path.join(path, "model.safetensors"))

            loaded = Eagle3Speculator.from_pretrained(path)

        self.assertEqual(loaded.hidden_size, 4)
        self.assertTrue(loaded.midlayer.norm_before_residual)
        self.assertTrue(torch.equal(loaded.midlayer.self_attn.q_proj.weight, source.midlayer.self_attn.q_proj.weight))

    def load_qwen3_speculator(self):
        return Eagle3Speculator.from_pretrained(
            SPECULATIVE_MODEL_PATH,
            target_model_path=TARGET_MODEL_PATH,
        ).float().eval()

    @requires_eagle3_models
    def test_loads_qwen3_specforge_weights_and_maps_logits(self):
        model = self.load_qwen3_speculator()

        input_ids = torch.tensor([[151644]], dtype=torch.long)
        positions = torch.tensor([[0]], dtype=torch.long)
        aux_hidden = torch.randn(1, 1, 3 * model.target_hidden_size)

        with torch.inference_mode():
            hidden_states, draft_logits, past_kv = model(input_ids, aux_hidden, positions)
            mapped_logits = model.compute_logits(hidden_states)

        num_kv_heads = model.midlayer.self_attn.num_kv_heads
        head_dim = model.midlayer.self_attn.head_dim
        self.assertEqual(hidden_states.shape, (1, 1, model.hidden_size))
        self.assertEqual(draft_logits.shape, (1, 1, model.draft_vocab_size))
        self.assertEqual(mapped_logits.shape, (1, 1, model.target_vocab_size))
        self.assertEqual(past_kv[0].shape, (1, num_kv_heads, 1, head_dim))
        self.assertEqual(past_kv[1].shape, (1, num_kv_heads, 1, head_dim))
        target_indices = torch.arange(model.draft_vocab_size) + model.d2t.cpu()
        self.assertTrue(torch.allclose(mapped_logits[0, 0, target_indices], draft_logits[0, 0]))
        self.assertEqual(int(model.t2d.sum().item()), model.draft_vocab_size)
        self.assertFalse(torch.isnan(draft_logits).any())
        self.assertFalse(torch.isnan(mapped_logits[torch.isfinite(mapped_logits)]).any())

    @requires_eagle3_models
    def test_proposes_first_target_vocab_token_from_aux_hidden(self):
        model = self.load_qwen3_speculator()

        input_ids = torch.tensor([[151644]], dtype=torch.long)
        positions = torch.tensor([[0]], dtype=torch.long)
        aux_hidden = torch.randn(1, 1, 3 * model.target_hidden_size)

        torch.manual_seed(0)
        with torch.inference_mode():
            output = model.propose(input_ids, aux_hidden, positions, temperature=1.0)

        self.assertEqual(output.hidden_states.shape, (1, 1, model.hidden_size))
        self.assertEqual(output.draft_logits.shape, (1, 1, model.draft_vocab_size))
        self.assertEqual(output.target_logits.shape, (1, 1, model.target_vocab_size))
        self.assertEqual(output.token_ids.shape, (1, 1))
        token_id = int(output.token_ids.item())
        self.assertGreaterEqual(token_id, 0)
        self.assertLess(token_id, model.target_vocab_size)
        self.assertTrue(bool(model.t2d[token_id]))


if __name__ == "__main__":
    unittest.main()
