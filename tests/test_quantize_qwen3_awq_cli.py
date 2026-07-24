import tempfile
import unittest
from pathlib import Path

import torch
from unittest.mock import patch


class Qwen3AWQCliTests(unittest.TestCase):
    def test_cuda_peak_stats_use_current_device_without_explicit_argument(self):
        from llmserve.quantization.quantize_qwen3 import (
            _max_cuda_memory_allocated,
            _reset_cuda_peak_memory_stats,
        )

        device = torch.device("cuda:0")
        with (
            patch("torch.cuda.set_device") as set_device,
            patch("torch.cuda.reset_peak_memory_stats") as reset_stats,
            patch("torch.cuda.max_memory_allocated", return_value=123) as max_memory,
        ):
            _reset_cuda_peak_memory_stats(device)
            actual = _max_cuda_memory_allocated(device)

        self.assertEqual(actual, 123)
        self.assertEqual(set_device.call_count, 2)
        set_device.assert_called_with(device)
        reset_stats.assert_called_once_with()
        max_memory.assert_called_once_with()

    def test_plain_text_calibration_loader_uses_nonempty_lines(self):
        from llmserve.quantization.quantize_qwen3 import load_calibration_texts

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "calibration.txt"
            path.write_text("first sample\n\n second sample \nthird\n", encoding="utf-8")
            texts = load_calibration_texts(path, max_samples=2)
        self.assertEqual(texts, ["first sample", "second sample"])

    def test_parser_exposes_reproducible_calibration_controls(self):
        from llmserve.quantization.quantize_qwen3 import build_parser

        args = build_parser().parse_args(
            [
                "--model",
                "/models/Qwen3-8B",
                "--output",
                "/models/Qwen3-8B-LLMServe-AWQ",
                "--calib-file",
                "/data/calibration.txt",
            ]
        )
        self.assertEqual(args.group_size, 128)
        self.assertEqual(args.n_grid, 20)
        self.assertEqual(args.search_tokens, 512)
        self.assertFalse(args.no_clip)

    def test_first_layer_capture_matches_transformers_qwen3_signature(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from llmserve.quantization.quantize_qwen3 import _capture_first_layer_input

        config = Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=32,
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

        self.assertEqual(hidden_states.shape, (1, 4, 16))
        self.assertIn("position_embeddings", kwargs)
        output = model.model.layers[0](hidden_states, **kwargs)
        self.assertEqual(output.shape, hidden_states.shape)


if __name__ == "__main__":
    unittest.main()
