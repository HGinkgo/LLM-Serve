import unittest
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 8, bias=False)
        self.v_proj = nn.Linear(8, 8, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)

    def forward(self, hidden_states):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        return self.o_proj(v + 0 * (q + k))


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(8, 16, bias=False)
        self.up_proj = nn.Linear(8, 16, bias=False)
        self.down_proj = nn.Linear(16, 8, bias=False)

    def forward(self, hidden_states):
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


class TinyQwen3Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(8, eps=1e-6)
        self.self_attn = TinyAttention()
        self.post_attention_layernorm = nn.RMSNorm(8, eps=1e-6)
        self.mlp = TinyMLP()

    def forward(self, hidden_states, **kwargs):
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states))
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class TinyQwen3Model(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([layer])
        self.model.embed_tokens = nn.Embedding(32, 8)
        self.model.norm = nn.RMSNorm(8)
        self.lm_head = nn.Linear(8, 32, bias=False)


class Qwen3LayerCalibrationTests(unittest.TestCase):
    def test_layer_calibration_packs_all_qwen3_projection_weights(self):
        from llmserve.quantization.qwen3_awq import calibrate_qwen3_layer

        torch.manual_seed(37)
        layer = TinyQwen3Layer()
        hidden_states = torch.randn(2, 4, 8)

        result = calibrate_qwen3_layer(
            layer,
            hidden_states,
            group_size=8,
            n_grid=4,
            max_tokens=16,
            apply_clip=True,
        )

        expected_names = {
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        }
        self.assertEqual(set(result.packed_weights), expected_names)
        self.assertEqual(result.hidden_states.shape, hidden_states.shape)
        self.assertTrue(torch.isfinite(result.hidden_states).all())
        self.assertGreater(result.evaluated_clip_group_count, 0)
        self.assertGreaterEqual(result.clipped_group_count, 0)

        for name, packed in result.packed_weights.items():
            module = layer.get_submodule(name)
            self.assertEqual(packed.qweight.shape[0], module.weight.shape[1])
            self.assertEqual(packed.scales.shape[1], module.weight.shape[0])

    def test_layer_cache_round_trip_reproduces_quantized_output(self):
        from llmserve.quantization.qwen3_awq import (
            apply_qwen3_layer_cache,
            calibrate_qwen3_layer,
            load_qwen3_layer_cache,
            save_qwen3_layer_cache,
        )
        from tempfile import TemporaryDirectory

        torch.manual_seed(41)
        source = TinyQwen3Layer()
        hidden_states = torch.randn(2, 4, 8)
        result = calibrate_qwen3_layer(
            source,
            hidden_states,
            group_size=8,
            n_grid=2,
            max_tokens=16,
            apply_clip=False,
        )

        with TemporaryDirectory() as temp_dir:
            cache_path = f"{temp_dir}/layer-0.safetensors"
            save_qwen3_layer_cache(source, result.packed_weights, cache_path)
            cached = load_qwen3_layer_cache(cache_path)
            restored = TinyQwen3Layer()
            apply_qwen3_layer_cache(restored, cached, group_size=8)
            restored_output = restored(hidden_states)

        torch.testing.assert_close(
            restored_output,
            result.hidden_states,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_checkpoint_export_uses_runtime_native_awq_names(self):
        from llmserve.quantization.qwen3_awq import (
            calibrate_qwen3_layer,
            export_qwen3_awq_checkpoint,
            save_qwen3_layer_cache,
        )
        from safetensors.torch import load_file
        from tempfile import TemporaryDirectory

        torch.manual_seed(43)
        layer = TinyQwen3Layer()
        model = TinyQwen3Model(layer)
        result = calibrate_qwen3_layer(
            layer,
            torch.randn(1, 4, 8),
            group_size=8,
            n_grid=2,
            max_tokens=8,
            apply_clip=False,
        )

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            (source / "config.json").write_text(
                json.dumps({"model_type": "qwen3", "torch_dtype": "bfloat16"}),
                encoding="utf-8",
            )
            (source / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            cache_path = root / "layer-0.safetensors"
            save_qwen3_layer_cache(layer, result.packed_weights, cache_path)

            export_qwen3_awq_checkpoint(
                model,
                [cache_path],
                source,
                output,
                calibration_metadata={"samples": 1},
                group_size=8,
                max_shard_size="1MB",
            )

            tensors = load_file(output / "model.safetensors")
            prefix = "model.layers.0.self_attn.q_proj"
            self.assertNotIn(f"{prefix}.weight", tensors)
            self.assertIn(f"{prefix}.qweight", tensors)
            self.assertIn(f"{prefix}.qzeros", tensors)
            self.assertIn(f"{prefix}.scales", tensors)
            self.assertIn("model.embed_tokens.weight", tensors)
            config = json.loads((output / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["quantization_config"]["quant_method"], "awq")
            self.assertEqual(config["quantization_config"]["group_size"], 8)
            manifest = json.loads(
                (output / "quantization_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["calibration"]["samples"], 1)
            self.assertEqual(manifest["format"], "autoawq-gemm")


if __name__ == "__main__":
    unittest.main()
