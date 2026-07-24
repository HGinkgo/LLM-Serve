import unittest
import tempfile
from dataclasses import replace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from torch import nn

from llmserve.layers.quantization.awq import (
    AWQRuntimeConfig,
    awq_reference_linear,
)


AWQ_PACK_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)


def pack_awq_int4(values: torch.Tensor) -> torch.Tensor:
    chunks = values.to(torch.int64).view(values.shape[0], -1, 8)
    chunks = chunks[:, :, AWQ_PACK_ORDER]
    shifts = torch.arange(0, 32, 4, dtype=torch.int64)
    return torch.sum((chunks & 0xF) << shifts, dim=-1).to(torch.int32)


def make_awq_config() -> AWQRuntimeConfig:
    return AWQRuntimeConfig(
        bits=4,
        group_size=128,
        zero_point=True,
        version="gemm",
        backend="autoawq",
        activation_dtype=torch.bfloat16,
    )


class AWQLinearBackendTests(unittest.TestCase):
    def setUp(self):
        rank = patch("llmserve.layers.linear.dist.get_rank", return_value=0)
        world_size = patch(
            "llmserve.layers.linear.dist.get_world_size", return_value=1
        )
        self.addCleanup(rank.stop)
        self.addCleanup(world_size.stop)
        rank.start()
        world_size.start()

    def test_column_linear_creates_checkpoint_native_parameters(self):
        from llmserve.layers.linear import ColumnParallelLinear

        layer = ColumnParallelLinear(128, 16, quant_config=make_awq_config())

        self.assertFalse(hasattr(layer, "weight"))
        self.assertEqual(layer.qweight.shape, (128, 2))
        self.assertEqual(layer.qzeros.shape, (1, 2))
        self.assertEqual(layer.scales.shape, (1, 16))
        self.assertEqual(layer.qweight.dtype, torch.int32)
        self.assertEqual(layer.qzeros.dtype, torch.int32)
        self.assertEqual(layer.scales.dtype, torch.bfloat16)
        self.assertFalse(layer.qweight.requires_grad)

    def test_merged_column_loader_slices_packed_output_dimension(self):
        from llmserve.layers.linear import MergedColumnParallelLinear

        layer = MergedColumnParallelLinear(
            128,
            [16, 8],
            quant_config=make_awq_config(),
        )
        shard_shapes = {
            "qweight": [(128, 2), (128, 1)],
            "qzeros": [(1, 2), (1, 1)],
            "scales": [(1, 16), (1, 8)],
        }

        for name, shapes in shard_shapes.items():
            param = getattr(layer, name)
            first = torch.full(shapes[0], 11, dtype=param.dtype)
            second = torch.full(shapes[1], 22, dtype=param.dtype)
            param.weight_loader(param, first, 0)
            param.weight_loader(param, second, 1)
            expected = torch.cat((first, second), dim=1)
            torch.testing.assert_close(param, expected, rtol=0, atol=0)

    def test_qkv_loader_slices_packed_output_dimension(self):
        from llmserve.layers.linear import QKVParallelLinear

        layer = QKVParallelLinear(
            hidden_size=128,
            head_size=8,
            total_num_heads=2,
            total_num_kv_heads=1,
            quant_config=make_awq_config(),
        )
        logical_sizes = {"q": 16, "k": 8, "v": 8}

        for name in ("qweight", "qzeros", "scales"):
            param = getattr(layer, name)
            for value, shard_id in enumerate(("q", "k", "v"), start=1):
                output_size = logical_sizes[shard_id]
                packed_size = output_size // 8 if name != "scales" else output_size
                shape = list(param.shape)
                shape[1] = packed_size
                loaded = torch.full(shape, value, dtype=param.dtype)
                param.weight_loader(param, loaded, shard_id)
            expected = torch.cat(
                [
                    torch.full(
                        (param.shape[0], logical_sizes[shard_id] // 8),
                        value,
                        dtype=param.dtype,
                    )
                    if name != "scales"
                    else torch.full(
                        (param.shape[0], logical_sizes[shard_id]),
                        value,
                        dtype=param.dtype,
                    )
                    for value, shard_id in enumerate(("q", "k", "v"), start=1)
                ],
                dim=1,
            )
            torch.testing.assert_close(param, expected, rtol=0, atol=0)

    def test_awq_forward_matches_reference_backend(self):
        from llmserve.layers.linear import ColumnParallelLinear

        layer = ColumnParallelLinear(128, 16, quant_config=make_awq_config())
        qvalues = torch.arange(128 * 16, dtype=torch.int32).view(128, 16) % 16
        zeros = torch.full((1, 16), 8, dtype=torch.int32)
        scales = torch.linspace(0.01, 0.16, 16, dtype=torch.bfloat16).view(1, 16)
        qweight = pack_awq_int4(qvalues)
        qzeros = pack_awq_int4(zeros)
        layer.qweight.weight_loader(layer.qweight, qweight)
        layer.qzeros.weight_loader(layer.qzeros, qzeros)
        layer.scales.weight_loader(layer.scales, scales)
        inputs = torch.randn((4, 128), dtype=torch.bfloat16)

        actual = layer(inputs)
        expected = awq_reference_linear(
            inputs,
            qweight,
            qzeros,
            scales,
            group_size=128,
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_awq_linear_uses_triton_backend_for_supported_decode_shape(self):
        from llmserve.layers.linear import ColumnParallelLinear

        config = replace(make_awq_config(), execution_backend="triton")
        with torch.device("cuda"):
            layer = ColumnParallelLinear(128, 64, quant_config=config)
        qvalues = torch.arange(128 * 64, dtype=torch.int32).view(128, 64) % 16
        zeros = torch.full((1, 64), 8, dtype=torch.int32)
        scales = torch.linspace(0.01, 0.64, 64, dtype=torch.bfloat16).view(1, 64)
        layer.qweight.weight_loader(layer.qweight, pack_awq_int4(qvalues))
        layer.qzeros.weight_loader(layer.qzeros, pack_awq_int4(zeros))
        layer.scales.weight_loader(layer.scales, scales)
        inputs = torch.randn((4, 128), dtype=torch.bfloat16, device="cuda")

        actual = layer(inputs)
        expected = awq_reference_linear(
            inputs,
            layer.qweight,
            layer.qzeros,
            layer.scales,
            group_size=128,
        )

        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.05)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_awq_linear_replaces_checkpoint_packing_for_cuda_backend(self):
        from llmserve.layers.linear import ColumnParallelLinear

        config = replace(make_awq_config(), execution_backend="cuda")
        with torch.device("cuda"):
            layer = ColumnParallelLinear(128, 64, quant_config=config)
        qvalues = torch.arange(128 * 64, dtype=torch.int32).view(128, 64) % 16
        zeros = torch.full((1, 64), 8, dtype=torch.int32)
        scales = torch.linspace(0.01, 0.64, 64, dtype=torch.bfloat16).view(1, 64)
        qweight = pack_awq_int4(qvalues).cuda()
        qzeros = pack_awq_int4(zeros).cuda()
        layer.qweight.weight_loader(layer.qweight, qweight)
        layer.qzeros.weight_loader(layer.qzeros, qzeros)
        layer.scales.weight_loader(layer.scales, scales)
        inputs = torch.randn((4, 128), dtype=torch.bfloat16, device="cuda")
        expected = awq_reference_linear(
            inputs,
            qweight,
            qzeros,
            layer.scales,
            group_size=128,
        )

        layer.quant_method.process_weights_after_loading(layer)
        actual = layer(inputs)

        self.assertFalse(hasattr(layer, "qweight"))
        self.assertFalse(hasattr(layer, "qzeros"))
        self.assertEqual(layer.awq_qweight.shape, (8, 1, 4, 32))
        self.assertEqual(layer.awq_qzeros.shape, (1, 8))
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.05)

    def test_awq_linear_rejects_tensor_parallel_construction(self):
        from llmserve.layers.linear import ColumnParallelLinear

        with patch(
            "llmserve.layers.linear.dist.get_world_size", return_value=2
        ), self.assertRaisesRegex(ValueError, "tensor parallel"):
            ColumnParallelLinear(128, 16, quant_config=make_awq_config())

    def test_merged_awq_linear_rejects_individually_unpacked_shards(self):
        from llmserve.layers.linear import (
            MergedColumnParallelLinear,
            QKVParallelLinear,
        )

        with self.assertRaisesRegex(ValueError, "shard.*pack_factor"):
            MergedColumnParallelLinear(
                128,
                [12, 12],
                quant_config=make_awq_config(),
            )
        with self.assertRaisesRegex(ValueError, "shard.*pack_factor"):
            QKVParallelLinear(
                hidden_size=128,
                head_size=4,
                total_num_heads=4,
                total_num_kv_heads=1,
                quant_config=make_awq_config(),
            )

    def test_qwen3_threads_awq_backend_through_all_target_projections(self):
        from transformers import Qwen3Config

        from llmserve.models.qwen3 import Qwen3ForCausalLM

        model_config = Qwen3Config(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=128,
            attention_bias=False,
            tie_word_embeddings=False,
        )

        model = Qwen3ForCausalLM(
            model_config,
            quant_config=make_awq_config(),
        )

        layer = model.model.layers[0]
        projections = (
            layer.self_attn.qkv_proj,
            layer.self_attn.o_proj,
            layer.mlp.gate_up_proj,
            layer.mlp.down_proj,
        )
        self.assertTrue(all(hasattr(projection, "qweight") for projection in projections))
        self.assertTrue(hasattr(model.model.embed_tokens, "weight"))
        self.assertTrue(hasattr(model.lm_head, "weight"))

    def test_safetensors_loader_merges_awq_qkv_parameters(self):
        from llmserve.layers.linear import QKVParallelLinear
        from llmserve.utils.loader import load_model

        class ToyModel(nn.Module):
            packed_modules_mapping = {
                "q_proj": ("qkv_proj", "q"),
                "k_proj": ("qkv_proj", "k"),
                "v_proj": ("qkv_proj", "v"),
            }

            def __init__(self):
                super().__init__()
                self.qkv_proj = QKVParallelLinear(
                    hidden_size=128,
                    head_size=8,
                    total_num_heads=2,
                    total_num_kv_heads=1,
                    quant_config=make_awq_config(),
                )

        tensors = {}
        logical_sizes = {"q": 16, "k": 8, "v": 8}
        for value, shard_id in enumerate(("q", "k", "v"), start=1):
            output_size = logical_sizes[shard_id]
            tensors[f"{shard_id}_proj.qweight"] = torch.full(
                (128, output_size // 8), value, dtype=torch.int32
            )
            tensors[f"{shard_id}_proj.qzeros"] = torch.full(
                (1, output_size // 8), value + 3, dtype=torch.int32
            )
            tensors[f"{shard_id}_proj.scales"] = torch.full(
                (1, output_size), value + 6, dtype=torch.bfloat16
            )

        model = ToyModel()
        with tempfile.TemporaryDirectory() as model_dir:
            save_file(tensors, f"{model_dir}/model.safetensors")
            load_model(model, model_dir)

        for name, base_value in (("qweight", 1), ("qzeros", 4), ("scales", 7)):
            param = getattr(model.qkv_proj, name)
            expected_parts = []
            for offset, shard_id in enumerate(("q", "k", "v")):
                output_size = logical_sizes[shard_id]
                physical_size = output_size // 8 if name != "scales" else output_size
                expected_parts.append(
                    torch.full(
                        (param.shape[0], physical_size),
                        base_value + offset,
                        dtype=param.dtype,
                    )
                )
            expected = torch.cat(expected_parts, dim=1)
            torch.testing.assert_close(param, expected, rtol=0, atol=0)

    def test_safetensors_loader_rejects_missing_awq_qkv_shard(self):
        from llmserve.layers.linear import QKVParallelLinear
        from llmserve.utils.loader import load_model

        class ToyModel(nn.Module):
            packed_modules_mapping = {
                "q_proj": ("qkv_proj", "q"),
                "k_proj": ("qkv_proj", "k"),
                "v_proj": ("qkv_proj", "v"),
            }

            def __init__(self):
                super().__init__()
                self.qkv_proj = QKVParallelLinear(
                    hidden_size=128,
                    head_size=8,
                    total_num_heads=2,
                    total_num_kv_heads=1,
                    quant_config=make_awq_config(),
                )

        tensors = {}
        for value, (shard_id, output_size) in enumerate(
            (("q", 16), ("k", 8)),
            start=1,
        ):
            tensors[f"{shard_id}_proj.qweight"] = torch.full(
                (128, output_size // 8), value, dtype=torch.int32
            )
            tensors[f"{shard_id}_proj.qzeros"] = torch.full(
                (1, output_size // 8), value, dtype=torch.int32
            )
            tensors[f"{shard_id}_proj.scales"] = torch.full(
                (1, output_size), value, dtype=torch.bfloat16
            )

        with tempfile.TemporaryDirectory() as model_dir:
            save_file(tensors, f"{model_dir}/model.safetensors")
            with self.assertRaisesRegex(RuntimeError, r"missing.*v"):
                load_model(ToyModel(), model_dir)


if __name__ == "__main__":
    unittest.main()
