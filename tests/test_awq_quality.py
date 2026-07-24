import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from unittest.mock import patch


class AWQQualityReferenceTests(unittest.TestCase):
    def test_generation_uses_left_padding_for_decoder_only_model(self):
        from llmserve.quantization.awq_quality import _generate

        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 1
            padding_side = "right"

            def __call__(self, texts, **kwargs):
                self.assertEqual(self.padding_side, "left")
                self.assertEqual(texts, ["first", "second"])
                self.assertEqual(kwargs, {"return_tensors": "pt", "padding": True})
                return {
                    "input_ids": torch.tensor([[2, 3], [4, 0]]),
                    "attention_mask": torch.tensor([[1, 1], [1, 0]]),
                }

            def decode(self, output, skip_special_tokens):
                self.assertTrue(skip_special_tokens)
                return str(output.tolist())

            def assertEqual(self, actual, expected):
                unittest.TestCase().assertEqual(actual, expected)

            def assertTrue(self, value):
                unittest.TestCase().assertTrue(value)

        class Model:
            def generate(self, **kwargs):
                self.last_kwargs = kwargs
                return torch.tensor([[2, 3, 4], [4, 0, 5]])

        tokenizer = Tokenizer()
        model = Model()
        result = _generate(
            model,
            tokenizer,
            ["first", "second"],
            torch.device("cpu"),
            max_new_tokens=1,
        )

        self.assertEqual(tokenizer.padding_side, "left")
        self.assertEqual(len(result), 2)
        self.assertEqual(model.last_kwargs["max_new_tokens"], 1)

    def test_quality_cuda_stats_do_not_pass_explicit_visible_device(self):
        from llmserve.quantization.awq_quality import (
            _max_cuda_memory_allocated,
            _reset_cuda_peak_memory_stats,
        )

        device = torch.device("cuda:0")
        with (
            patch("torch.cuda.set_device") as set_device,
            patch("torch.cuda.reset_peak_memory_stats") as reset_stats,
            patch("torch.cuda.max_memory_allocated", return_value=456) as max_memory,
        ):
            _reset_cuda_peak_memory_stats(device)
            actual = _max_cuda_memory_allocated(device)

        self.assertEqual(actual, 456)
        self.assertEqual(set_device.call_count, 2)
        reset_stats.assert_called_once_with()
        max_memory.assert_called_once_with()

    def test_reference_linear_matches_packed_weight_dequantization(self):
        from llmserve.quantization.awq_calibration import quantize_awq_weight
        from llmserve.quantization.awq_quality import ReferenceAWQLinear

        torch.manual_seed(47)
        weight = torch.randn(16, 128, dtype=torch.float32)
        inputs = torch.randn(3, 128, dtype=torch.float32)
        packed = quantize_awq_weight(weight, group_size=128)
        linear = ReferenceAWQLinear(128, 16, group_size=128, dtype=torch.float32)
        linear.qweight.copy_(packed.qweight)
        linear.qzeros.copy_(packed.qzeros)
        linear.scales.copy_(packed.scales)

        actual = linear(inputs)
        expected_weight = (
            (torch.round(weight.reshape(-1, 128) / packed.scales.t().reshape(-1, 1)))
        )
        self.assertEqual(actual.shape, (3, 16))
        self.assertTrue(torch.isfinite(expected_weight).all())
        from llmserve.layers.quantization.awq import dequantize_awq_gemm
        dequantized = dequantize_awq_gemm(
            packed.qweight,
            packed.qzeros,
            packed.scales,
            group_size=128,
        ).t()
        torch.testing.assert_close(actual, F.linear(inputs, dequantized))

    def test_quality_parser_requires_model_eval_file_and_output(self):
        from llmserve.quantization.awq_quality import build_parser

        args = build_parser().parse_args([
            "--model", "/models/Qwen3-8B-AWQ",
            "--eval-file", "/data/eval.txt",
            "--output", "/results/quality.json",
        ])
        self.assertEqual(args.mode, "auto")
        self.assertEqual(args.sequence_length, 128)
        self.assertEqual(args.max_eval_tokens, 2048)

    def test_exported_tiny_qwen3_loads_through_reference_quality_model(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from llmserve.quantization.awq_quality import _load_awq_reference_model
        from llmserve.quantization.qwen3_awq import (
            calibrate_qwen3_layer,
            export_qwen3_awq_checkpoint,
            save_qwen3_layer_cache,
        )
        from llmserve.quantization.quantize_qwen3 import _capture_first_layer_input

        config = Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=32,
            tie_word_embeddings=False,
        )
        model = Qwen3ForCausalLM(config).eval()
        encoded = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
        }
        hidden_states, kwargs = _capture_first_layer_input(
            model,
            encoded,
            torch.device("cpu"),
        )
        result = calibrate_qwen3_layer(
            model.model.layers[0],
            hidden_states,
            forward_kwargs=kwargs,
            group_size=8,
            n_grid=1,
            max_tokens=8,
            apply_clip=False,
        )

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            config.to_json_file(source / "config.json")
            cache_path = root / "layer-0.safetensors"
            save_qwen3_layer_cache(
                model.model.layers[0],
                result.packed_weights,
                cache_path,
            )
            export_qwen3_awq_checkpoint(
                model,
                [cache_path],
                source,
                output,
                calibration_metadata={"test": True},
                group_size=8,
                max_shard_size="1MB",
            )
            restored = _load_awq_reference_model(output, torch.device("cpu"))
            logits = restored(**encoded, use_cache=False).logits

        self.assertEqual(logits.shape, (1, 4, 64))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
