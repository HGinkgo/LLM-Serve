import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import save_file

from llmserve.quantization.gptq import GPTQConfig
from llmserve.models.qwen3_moe import ExpertRouter, Qwen3MoeSparseMoeBlock
from llmserve.config import Config
from llmserve.layers.quantized import (
    GPTQLinear,
    gptq_qzeros_to_tinygemm_offset,
    marlin_permute_scales,
    unpack_gptq_qweight,
)
from llmserve.engine.model_runner import select_model_class
from llmserve.models.qwen3 import Qwen3ForCausalLM
from llmserve.models.qwen3_moe import Qwen3MoeForCausalLM
from llmserve.utils.loader import load_model


class GPTQConfigTest(unittest.TestCase):

    def test_accepts_the_official_qwen_gptq_int4_contract(self):
        config = GPTQConfig.from_dict({
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "desc_act": False,
            "checkpoint_format": "gptq",
        })

        self.assertEqual(config.bits, 4)
        self.assertEqual(config.group_size, 128)
        self.assertTrue(config.sym)

    def test_rejects_a_different_gptq_weight_layout(self):
        with self.assertRaisesRegex(ValueError, "GPTQ-Int4"):
            GPTQConfig.from_dict({
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 64,
                "sym": True,
                "desc_act": False,
                "checkpoint_format": "gptq",
            })

    def test_config_allows_only_single_gpu_qwen3_moe_gptq(self):
        hf_config = SimpleNamespace(
            model_type="qwen3_moe",
            max_position_embeddings=4096,
            quantization_config={
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 128,
                "sym": True,
                "desc_act": False,
                "checkpoint_format": "gptq",
            },
        )
        with tempfile.TemporaryDirectory() as model_dir, patch(
            "llmserve.config.AutoConfig.from_pretrained",
            return_value=hf_config,
        ):
            config = Config(
                model=model_dir,
                tensor_parallel_size=1,
                enforce_eager=True,
            )

        self.assertIsInstance(config.quantization, GPTQConfig)

    def test_config_rejects_gptq_moe_tensor_parallelism(self):
        hf_config = SimpleNamespace(
            model_type="qwen3_moe",
            max_position_embeddings=4096,
            quantization_config={
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 128,
                "sym": True,
                "desc_act": False,
                "checkpoint_format": "gptq",
            },
        )
        with tempfile.TemporaryDirectory() as model_dir, patch(
            "llmserve.config.AutoConfig.from_pretrained",
            return_value=hf_config,
        ), self.assertRaisesRegex(ValueError, "tensor_parallel_size=1"):
            Config(model=model_dir, tensor_parallel_size=2, enforce_eager=True)

    def test_config_rejects_unquantized_moe_checkpoints(self):
        hf_config = SimpleNamespace(
            model_type="qwen3_moe",
            max_position_embeddings=4096,
            quantization_config=None,
        )
        with tempfile.TemporaryDirectory() as model_dir, patch(
            "llmserve.config.AutoConfig.from_pretrained",
            return_value=hf_config,
        ), self.assertRaisesRegex(ValueError, "GPTQ-Int4"):
            Config(model=model_dir, enforce_eager=True)

    def test_config_requires_eager_execution_for_gptq_moe(self):
        hf_config = SimpleNamespace(
            model_type="qwen3_moe",
            max_position_embeddings=4096,
            quantization_config={
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 128,
                "sym": True,
                "desc_act": False,
                "checkpoint_format": "gptq",
            },
        )
        with tempfile.TemporaryDirectory() as model_dir, patch(
            "llmserve.config.AutoConfig.from_pretrained",
            return_value=hf_config,
        ), self.assertRaisesRegex(ValueError, "enforce_eager=True"):
            Config(model=model_dir)

    def test_config_requires_a_provider_library_for_marlin(self):
        hf_config = SimpleNamespace(
            model_type="qwen3_moe",
            max_position_embeddings=4096,
            quantization_config={
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 128,
                "sym": True,
                "desc_act": False,
                "checkpoint_format": "gptq",
            },
        )
        with tempfile.TemporaryDirectory() as model_dir, patch(
            "llmserve.config.AutoConfig.from_pretrained",
            return_value=hf_config,
        ), self.assertRaisesRegex(ValueError, "marlin_library"):
            Config(
                model=model_dir,
                enforce_eager=True,
                gptq_backend="marlin",
            )

    def test_config_rejects_speculative_decoding_for_gptq_moe(self):
        hf_config = SimpleNamespace(
            model_type="qwen3_moe",
            max_position_embeddings=4096,
            quantization_config={
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 128,
                "sym": True,
                "desc_act": False,
                "checkpoint_format": "gptq",
            },
        )
        with tempfile.TemporaryDirectory() as model_dir, tempfile.TemporaryDirectory() as draft_dir, patch(
            "llmserve.config.AutoConfig.from_pretrained",
            return_value=hf_config,
        ), self.assertRaisesRegex(ValueError, "does not support speculative decoding"):
            Config(
                model=model_dir,
                enforce_eager=True,
                speculative_model=draft_dir,
            )

    def test_runner_selects_moe_only_for_a_validated_quantized_config(self):
        self.assertIs(
            select_model_class(SimpleNamespace(quantization=None)),
            Qwen3ForCausalLM,
        )
        self.assertIs(
            select_model_class(SimpleNamespace(quantization=object())),
            Qwen3MoeForCausalLM,
        )


class ExpertRouterTest(unittest.TestCase):

    def test_selects_topk_and_normalizes_selected_probabilities(self):
        router = ExpertRouter(num_experts=4, top_k=2, normalize_topk=True)
        router_weight = torch.tensor([
            [2.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [0.0, 1.0],
        ])
        hidden_states = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        weights, expert_ids = router(hidden_states, router_weight)

        self.assertEqual(expert_ids.tolist(), [[0, 1], [2, 3]])
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(2)))
        self.assertGreater(weights[0, 0], weights[0, 1])
        self.assertGreater(weights[1, 0], weights[1, 1])

class SparseMoeBlockTest(unittest.TestCase):

    def test_groups_assignments_and_combines_each_active_expert_once(self):
        config = SimpleNamespace(
            hidden_size=128,
            moe_intermediate_size=128,
            num_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            quantization_config={"group_size": 128},
        )
        block = Qwen3MoeSparseMoeBlock(config)

        class ScaleExpert(nn.Module):
            def __init__(self, scale):
                super().__init__()
                self.scale = scale

            def forward(self, hidden_states):
                return hidden_states * self.scale

        block.experts = nn.ModuleList([ScaleExpert(scale) for scale in (2.0, 3.0, 5.0, 7.0)])
        block.gate.weight.data.zero_()
        block.gate.weight.data[0, 0] = 2.0
        block.gate.weight.data[1, 0] = 1.0
        hidden_states = torch.zeros(2, 128)
        hidden_states[:, 0] = 1.0

        output = block(hidden_states)
        weights, expert_ids = block.router(hidden_states, block.gate.weight)
        expected_scale = torch.zeros(2)
        for token_id in range(2):
            for weight, expert_id in zip(weights[token_id], expert_ids[token_id]):
                expected_scale[token_id] += weight * (2.0, 3.0, 5.0, 7.0)[expert_id]

        self.assertTrue(torch.allclose(output[:, 0], expected_scale))
        self.assertTrue(torch.equal(output[:, 1:], torch.zeros(2, 127)))


class GPTQLinearTest(unittest.TestCase):

    def test_marlin_scale_permutation_uses_the_single_group_kernel_layout(self):
        scales = torch.arange(256, dtype=torch.float16).reshape(1, 256)

        actual = marlin_permute_scales(
            scales,
            input_size=128,
            output_size=256,
            group_size=128,
        )

        permutation = torch.tensor([
            2 * index + offset
            for index in range(4)
            for offset in (0, 1, 8, 9, 16, 17, 24, 25)
        ])
        expected = scales.reshape(-1, 32).index_select(1, permutation).reshape(1, 256)
        self.assertTrue(torch.equal(actual, expected))


    def test_unpacks_columns_and_gptq_zero_points_before_matmul(self):
        linear = GPTQLinear(8, 8, group_size=4)
        linear.qweight.data.copy_(torch.tensor([[0x22221111] * 8]))
        # GPTQ stores zero_point - 1 in qzeros, packed along output features.
        linear.qzeros.data.copy_(torch.tensor([[0], [0x11111111]]))
        linear.scales.data.copy_(torch.tensor([
            [2.0] * 8,
            [3.0] * 8,
        ]))

        weight = linear.dequantize_weight()

        self.assertTrue(torch.equal(weight, torch.zeros(8, 8)))
        self.assertTrue(torch.equal(
            linear(torch.ones(1, 8)),
            weight.sum(dim=0, keepdim=True),
        ))

    def test_unpacks_gptq_input_packing_into_input_output_codes(self):
        packed = torch.tensor([
            [0x76543210, -0x01234568],
            [0x76543210, -0x01234568],
        ], dtype=torch.int32)

        codes = unpack_gptq_qweight(packed, input_size=16, output_size=2)

        self.assertEqual(codes.tolist(), [
            *[[value, value + 8] for value in range(8)],
            *[[value, value + 8] for value in range(8)],
        ])

    def test_converts_gptq_zero_points_to_tinygemm_offsets(self):
        scales = torch.tensor([[0.5, 0.25], [1.0, 2.0]])
        zero_points = torch.tensor([[8, 7], [9, 8]], dtype=torch.int32)

        offsets = gptq_qzeros_to_tinygemm_offset(scales, zero_points)

        self.assertTrue(torch.equal(offsets, torch.tensor([
            [0.0, 0.25],
            [-1.0, 0.0],
        ])))

    def test_rejects_non_contiguous_gptq_group_mapping(self):
        linear = GPTQLinear(8, 8, group_size=4)
        linear.g_idx.data.copy_(torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.int32))

        with self.assertRaisesRegex(ValueError, "contiguous groups"):
            linear.validate_group_mapping()

    def test_reference_path_preserves_the_input_dtype(self):
        linear = GPTQLinear(8, 8, group_size=4)
        linear.qweight.data.fill_(0x11111111)
        linear.qzeros.data.zero_()
        linear.scales.data.fill_(1.0)

        output = linear(torch.ones(1, 8, dtype=torch.float16))

        self.assertEqual(output.dtype, torch.float16)

    def test_generic_safetensors_loader_accepts_all_gptq_parameters(self):
        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = GPTQLinear(8, 8, group_size=4)

        checkpoint = {
            "linear.qweight": torch.full((1, 8), 0x11111111, dtype=torch.int32),
            "linear.qzeros": torch.zeros((2, 1), dtype=torch.int32),
            "linear.scales": torch.ones((2, 8)),
            "linear.g_idx": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32),
        }
        with tempfile.TemporaryDirectory() as model_dir:
            save_file(checkpoint, f"{model_dir}/model.safetensors")
            module = Module()
            load_model(module, model_dir)

        self.assertTrue(torch.equal(module.linear.qweight, checkpoint["linear.qweight"]))
        self.assertTrue(torch.equal(module.linear.g_idx, checkpoint["linear.g_idx"]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_native_cuda_provider_matches_reference_and_releases_source_tensors(self):
        input_size = 128
        output_size = 16
        linear = GPTQLinear(input_size, output_size, group_size=128).cuda()
        codes = torch.randint(
            0,
            16,
            (input_size, output_size),
            dtype=torch.int32,
            device="cuda",
        )
        shifts = torch.arange(8, dtype=torch.int32, device="cuda") * 4
        linear.qweight.data.copy_(
            (codes.reshape(input_size // 8, 8, output_size) << shifts.view(1, 8, 1)).sum(dim=1)
        )
        linear.qzeros.data.fill_(0x77777777)
        linear.scales.data.uniform_(0.001, 0.1)
        hidden_states = torch.randn(3, input_size, dtype=torch.float16, device="cuda")

        reference = linear(hidden_states)
        linear.prepare_for_runtime()
        actual = linear(hidden_states)

        self.assertTrue(torch.allclose(actual, reference, rtol=2e-2, atol=2e-2))
        self.assertEqual(linear.qweight.numel(), 0)
        self.assertEqual(linear.qzeros.numel(), 0)
        self.assertEqual(linear.scales.numel(), 0)
        self.assertEqual(linear.g_idx.numel(), 0)

    @unittest.skipUnless(
        torch.cuda.is_available() and os.environ.get("LLMSERVE_TEST_MARLIN_LIBRARY"),
        "set LLMSERVE_TEST_MARLIN_LIBRARY and make CUDA available",
    )
    def test_marlin_cuda_provider_matches_reference_and_releases_source_tensors(self):
        linear = GPTQLinear(
            128,
            256,
            group_size=128,
            backend="marlin",
            marlin_library=os.environ["LLMSERVE_TEST_MARLIN_LIBRARY"],
        ).cuda()
        codes = torch.randint(
            0,
            16,
            (128, 256),
            dtype=torch.int32,
            device="cuda",
        )
        shifts = torch.arange(8, dtype=torch.int32, device="cuda") * 4
        linear.qweight.data.copy_(
            (codes.reshape(16, 8, 256) << shifts.view(1, 8, 1)).sum(dim=1)
        )
        linear.qzeros.data.fill_(0x77777777)
        linear.scales.data.uniform_(0.001, 0.1)
        hidden_states = torch.randn(3, 128, dtype=torch.float16, device="cuda")

        reference = linear(hidden_states)
        linear.prepare_for_runtime()
        actual = linear(hidden_states)

        self.assertTrue(torch.allclose(actual, reference, rtol=2e-2, atol=2e-2))
        self.assertEqual(linear.qweight.numel(), 0)
        self.assertEqual(linear.qzeros.numel(), 0)
        self.assertEqual(linear.scales.numel(), 0)
        self.assertEqual(linear.g_idx.numel(), 0)

    @unittest.skipUnless(
        torch.cuda.is_available() and os.environ.get("LLMSERVE_TEST_GPTQ_MOE_MODEL"),
        "set LLMSERVE_TEST_GPTQ_MOE_MODEL and make CUDA available",
    )
    def test_official_qwen_gptq_attention_projection_matches_native_provider(self):
        model_dir = Path(os.environ["LLMSERVE_TEST_GPTQ_MOE_MODEL"])
        model_file = model_dir / "model.safetensors"
        self.assertTrue(model_file.is_file(), f"missing checkpoint file: {model_file}")
        linear = GPTQLinear(2048, 4096, group_size=128).cuda()
        prefix = "model.layers.0.self_attn.q_proj"
        with safe_open(str(model_file), framework="pt", device="cpu") as checkpoint:
            linear.qweight.data.copy_(checkpoint.get_tensor(f"{prefix}.qweight"))
            linear.qzeros.data.copy_(checkpoint.get_tensor(f"{prefix}.qzeros"))
            linear.scales.data.copy_(checkpoint.get_tensor(f"{prefix}.scales"))
            linear.g_idx.data.copy_(checkpoint.get_tensor(f"{prefix}.g_idx"))
        hidden_states = torch.randn(3, 2048, dtype=torch.float16, device="cuda")

        reference = linear(hidden_states)
        linear.prepare_for_runtime()
        actual = linear(hidden_states)

        self.assertTrue(torch.allclose(actual, reference, rtol=2e-2, atol=2e-2))


if __name__ == "__main__":
    unittest.main()
