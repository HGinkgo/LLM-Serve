import unittest

import torch


class AWQLinearProfileTests(unittest.TestCase):
    def test_merge_projection_tensors_uses_checkpoint_output_dimension(self):
        from benchmarks.awq_linear_profile import merge_projection_tensors

        parts = []
        for value, output_size in ((1, 16), (2, 8), (3, 8)):
            parts.append(
                (
                    torch.full((128, output_size // 8), value, dtype=torch.int32),
                    torch.full((1, output_size // 8), value + 3, dtype=torch.int32),
                    torch.full((1, output_size), value + 6, dtype=torch.bfloat16),
                )
            )

        projection = merge_projection_tensors("qkv", parts)

        self.assertEqual(projection.name, "qkv")
        self.assertEqual(projection.qweight.shape, (128, 4))
        self.assertEqual(projection.qzeros.shape, (1, 4))
        self.assertEqual(projection.scales.shape, (1, 32))
        self.assertEqual(projection.output_size, 32)
        self.assertEqual(projection.input_size, 128)
        torch.testing.assert_close(
            projection.qweight[:, :2],
            torch.full((128, 2), 1, dtype=torch.int32),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            projection.scales[:, -8:],
            torch.full((1, 8), 9, dtype=torch.bfloat16),
            rtol=0,
            atol=0,
        )

    def test_merge_projection_tensors_rejects_incompatible_inputs(self):
        from benchmarks.awq_linear_profile import merge_projection_tensors

        first = (
            torch.zeros((128, 1), dtype=torch.int32),
            torch.zeros((1, 1), dtype=torch.int32),
            torch.ones((1, 8), dtype=torch.bfloat16),
        )
        second = (
            torch.zeros((256, 1), dtype=torch.int32),
            torch.zeros((2, 1), dtype=torch.int32),
            torch.ones((2, 8), dtype=torch.bfloat16),
        )

        with self.assertRaisesRegex(ValueError, "input/group"):
            merge_projection_tensors("invalid", [first, second])

    def test_summarize_stage_latencies_reports_dequantization_share(self):
        from benchmarks.awq_linear_profile import summarize_stage_latencies

        measurements = {
            "qkv": {
                "dequantize": 3.0,
                "matmul": 1.0,
                "reference": 4.5,
                "triton": 0.5,
                "cuda": 0.25,
            },
            "o": {
                "dequantize": 2.0,
                "matmul": 1.0,
                "reference": 3.5,
                "triton": 0.5,
                "cuda": 0.25,
            },
            "gate_up": {
                "dequantize": 6.0,
                "matmul": 2.0,
                "reference": 9.0,
                "triton": 0.5,
                "cuda": 0.25,
            },
            "down": {
                "dequantize": 4.0,
                "matmul": 1.0,
                "reference": 5.5,
                "triton": 0.5,
                "cuda": 0.25,
            },
        }

        summary = summarize_stage_latencies(
            measurements,
            num_hidden_layers=36,
        )

        self.assertEqual(summary["per_layer_ms"]["dequantize"], 15.0)
        self.assertEqual(summary["per_layer_ms"]["matmul"], 5.0)
        self.assertEqual(summary["per_layer_ms"]["reference"], 22.5)
        self.assertEqual(summary["per_layer_ms"]["triton"], 2.0)
        self.assertEqual(summary["per_layer_ms"]["cuda"], 1.0)
        self.assertEqual(summary["dequantize_component_share_pct"], 75.0)
        self.assertEqual(summary["model_reference_total_ms"], 810.0)
        self.assertEqual(summary["model_triton_total_ms"], 72.0)
        self.assertEqual(summary["triton_speedup_vs_reference"], 11.25)
        self.assertEqual(summary["model_cuda_total_ms"], 36.0)
        self.assertEqual(summary["cuda_speedup_vs_reference"], 22.5)

    def test_parser_defaults_to_decode_token_counts(self):
        from benchmarks.awq_linear_profile import build_parser

        args = build_parser().parse_args(["--model", "/models/Qwen3-8B-AWQ"])

        self.assertEqual(args.num_tokens, (1, 4, 8, 16))
        self.assertEqual(args.layer_id, 0)
        self.assertEqual(args.warmup, 5)
        self.assertEqual(args.repeats, 20)


if __name__ == "__main__":
    unittest.main()
