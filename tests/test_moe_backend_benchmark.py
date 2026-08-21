import unittest


class MoeBackendBenchmarkTest(unittest.TestCase):

    def test_vllm_marlin_baseline_provenance_is_explicit(self):
        from benchmarks.moe_baseline import (
            build_moe_experiment_metadata,
            build_vllm_marlin_baseline,
        )

        baseline = build_vllm_marlin_baseline(
            source="/opt/.reference-vllm-0.9.1/vllm/_C.abi3.so"
        )
        report_metadata = build_moe_experiment_metadata(
            optimization="moe_gate_up_fusion",
            enabled=True,
            baseline=baseline,
        )

        self.assertEqual(baseline["runtime"], "vllm")
        self.assertEqual(baseline["backend"], "gptq_marlin")
        self.assertEqual(baseline["runtime_version"], "0.9.1")
        self.assertTrue(report_metadata["optimization"]["enabled"])

    def test_parse_backends_deduplicates_and_rejects_unknown_values(self):
        from benchmarks.moe_backend import parse_backends

        self.assertEqual(parse_backends("tinygemm,marlin,tinygemm"), ("tinygemm", "marlin"))
        with self.assertRaisesRegex(ValueError, "unknown GPTQ backend"):
            parse_backends("tinygemm,awq")

    def test_build_engine_kwargs_forwards_explicit_marlin_provider(self):
        from benchmarks.moe_backend import build_engine_kwargs

        kwargs = build_engine_kwargs(
            {
                "max_model_len": 512,
                "max_num_batched_tokens": 128,
                "max_num_seqs": 4,
                "gpu_memory_utilization": 0.85,
                "marlin_library": "/opt/vllm/_C.abi3.so",
            },
            backend="marlin",
        )

        self.assertEqual(kwargs["gptq_backend"], "marlin")
        self.assertEqual(kwargs["marlin_library"], "/opt/vllm/_C.abi3.so")
        self.assertTrue(kwargs["enforce_eager"])
        self.assertFalse(kwargs["enable_moe_gate_up_fusion"])

    def test_build_engine_kwargs_can_enable_gate_up_fusion_for_ab(self):
        from benchmarks.moe_backend import build_engine_kwargs

        kwargs = build_engine_kwargs(
            {
                "max_model_len": 512,
                "max_num_batched_tokens": 128,
                "max_num_seqs": 4,
                "gpu_memory_utilization": 0.85,
                "enable_moe_gate_up_fusion": True,
            },
            backend="tinygemm",
        )

        self.assertTrue(kwargs["enable_moe_gate_up_fusion"])

    def test_summarize_batch_metrics_uses_last_token_for_tpot(self):
        from benchmarks.moe_backend import summarize_batch_metrics

        result = summarize_batch_metrics(
            [
                {
                    "arrival_time": 0.0,
                    "first_token_time": 0.2,
                    "finish_time": 1.0,
                    "output_tokens": 5,
                    "token_times": [0.2, 0.4, 0.6, 0.8, 0.9],
                    "success": True,
                }
            ],
            wall_time=1.0,
            peak_allocated_bytes=123,
            peak_reserved_bytes=456,
        )

        self.assertEqual(result["output_tokens"], 5)
        self.assertEqual(result["output_throughput"], 5.0)
        self.assertEqual(result["ttft_ms"]["p50"], 200.0)
        self.assertEqual(result["tpot_ms"]["p50"], 175.0)
        self.assertEqual(result["peak_allocated_bytes"], 123)
        self.assertEqual(result["peak_reserved_bytes"], 456)


if __name__ == "__main__":
    unittest.main()
