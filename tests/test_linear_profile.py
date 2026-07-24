import unittest
from types import SimpleNamespace

import torch


class LinearProfileTests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            hidden_size=4096,
            intermediate_size=12288,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            num_hidden_layers=36,
        )

    def test_qwen3_projection_shapes_match_merged_runtime_linears(self):
        from benchmarks.linear_profile import qwen3_projection_shapes

        shapes = qwen3_projection_shapes(self.config)

        self.assertEqual(
            [(shape.name, shape.input_size, shape.output_size) for shape in shapes],
            [
                ("qkv", 4096, 6144),
                ("o", 4096, 4096),
                ("gate_up", 4096, 24576),
                ("down", 12288, 4096),
            ],
        )

    def test_qwen3_projection_shapes_reject_invalid_head_partition(self):
        from benchmarks.linear_profile import qwen3_projection_shapes

        self.config.hidden_size = 4100
        self.config.head_dim = None

        with self.assertRaisesRegex(ValueError, "hidden_size"):
            qwen3_projection_shapes(self.config)

    def test_summarize_latencies_reports_projection_and_model_totals(self):
        from benchmarks.linear_profile import summarize_latencies

        summary = summarize_latencies(
            {
                "qkv": 1.0,
                "o": 1.0,
                "gate_up": 4.0,
                "down": 2.0,
            },
            num_hidden_layers=36,
        )

        self.assertEqual(summary["per_layer_total_ms"], 8.0)
        self.assertEqual(summary["model_projection_total_ms"], 288.0)
        self.assertEqual(summary["share_pct"]["gate_up"], 50.0)
        self.assertEqual(summary["share_pct"]["down"], 25.0)

    def test_summarize_latencies_requires_all_projection_measurements(self):
        from benchmarks.linear_profile import summarize_latencies

        with self.assertRaisesRegex(ValueError, "missing projection latency"):
            summarize_latencies({"qkv": 1.0}, num_hidden_layers=36)

    def test_profile_projection_shapes_keeps_num_tokens_distinct_from_batch(self):
        from benchmarks.linear_profile import (
            profile_projection_shapes,
            qwen3_projection_shapes,
        )

        calls = []

        def measure(shape, num_tokens):
            calls.append((shape.name, num_tokens))
            return {
                "qkv": 1.0,
                "o": 1.0,
                "gate_up": 4.0,
                "down": 2.0,
            }[shape.name] * num_tokens

        result = profile_projection_shapes(
            qwen3_projection_shapes(self.config),
            num_hidden_layers=36,
            num_tokens=(1, 4),
            measure_latency=measure,
        )

        self.assertEqual(len(calls), 8)
        self.assertEqual([item["num_tokens"] for item in result], [1, 4])
        self.assertEqual(result[0]["summary"]["per_layer_total_ms"], 8.0)
        self.assertEqual(result[1]["summary"]["per_layer_total_ms"], 32.0)

    def test_parse_num_tokens_rejects_duplicates_and_non_positive_values(self):
        from benchmarks.linear_profile import parse_num_tokens

        self.assertEqual(parse_num_tokens("1,4,8,16"), (1, 4, 8, 16))
        with self.assertRaisesRegex(ValueError, "unique"):
            parse_num_tokens("1,4,4")
        with self.assertRaisesRegex(ValueError, "positive"):
            parse_num_tokens("0,1")

    def test_profile_cli_uses_decode_token_counts_by_default(self):
        from benchmarks.linear_profile import build_parser

        args = build_parser().parse_args(["--model", "/models/Qwen3-8B"])

        self.assertEqual(args.num_tokens, (1, 4, 8, 16))
        self.assertEqual(args.dtype, "bfloat16")
        self.assertEqual(args.warmup, 10)
        self.assertEqual(args.repeats, 50)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_measure_dense_cuda_returns_positive_latency(self):
        from benchmarks.linear_profile import LinearShape, measure_dense_cuda

        latency_ms = measure_dense_cuda(
            LinearShape("tiny", input_size=16, output_size=32),
            num_tokens=2,
            dtype=torch.float16,
            device=torch.device("cuda"),
            warmup=1,
            repeats=2,
        )

        self.assertGreater(latency_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
