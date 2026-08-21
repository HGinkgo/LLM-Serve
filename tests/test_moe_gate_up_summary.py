from __future__ import annotations

import importlib.util
import unittest


def _report(*, fused: bool, result: dict) -> dict:
    return {
        "model": "/models/Qwen3-30B-A3B-GPTQ-Int4",
        "metadata": {
            "git_commit": "578feb19",
            "git_dirty": True,
        },
        "input": {
            "case_count": 1,
            "prompt_token_sha256": "fixed-trace",
            "cases": [{"id": "case", "prompt_token_count": 8}],
        },
        "runtime_config": {
            "max_model_len": 512,
            "max_num_batched_tokens": 128,
            "max_num_seqs": 4,
            "gpu_memory_utilization": 0.85,
            "seed": 20260819,
            "max_new_tokens": 16,
            "marlin_library": "/opt/.reference-vllm-0.9.1/vllm/_C.abi3.so",
            "enable_moe_gate_up_fusion": fused,
            "backends": ["marlin"],
            "batch_sizes": [1],
            "repeats": 2,
        },
        "results": [result],
    }


class MoeGateUpSummaryTest(unittest.TestCase):

    def _summarize(self):
        spec = importlib.util.find_spec("benchmarks.moe_gate_up_summary")
        self.assertIsNotNone(spec, "MoE A/B summary module is missing")
        from benchmarks.moe_gate_up_summary import summarize_ab_report

        return summarize_ab_report

    def test_aggregates_raw_request_metrics_and_reports_delta(self):
        summarize_ab_report = self._summarize()

        control = _report(
            fused=False,
            result={
                "backend": "marlin",
                "batch_size": 1,
                "repeat": 0,
                "request_count": 2,
                "completed": 2,
                "failed": 0,
                "output_tokens": 8,
                "wall_time_seconds": 4.0,
                "peak_allocated_bytes": 100,
                "peak_reserved_bytes": 120,
                "requests": [
                    {"output_tokens": 4, "ttft_ms": 100.0, "tpot_ms": 20.0},
                    {"output_tokens": 4, "ttft_ms": 300.0, "tpot_ms": 40.0},
                ],
            },
        )
        optimized = _report(
            fused=True,
            result={
                "backend": "marlin",
                "batch_size": 1,
                "repeat": 0,
                "request_count": 2,
                "completed": 2,
                "failed": 0,
                "output_tokens": 8,
                "wall_time_seconds": 2.0,
                "peak_allocated_bytes": 150,
                "peak_reserved_bytes": 170,
                "requests": [
                    {"output_tokens": 4, "ttft_ms": 50.0, "tpot_ms": 10.0},
                    {"output_tokens": 4, "ttft_ms": 150.0, "tpot_ms": 30.0},
                ],
            },
        )

        summary = summarize_ab_report(control=control, optimized=optimized)

        self.assertEqual(summary["external_baseline"]["runtime"], "vllm")
        self.assertEqual(summary["external_baseline"]["backend"], "gptq_marlin")
        self.assertEqual(summary["external_baseline"]["runtime_version"], "0.9.1")
        self.assertEqual(summary["workload"]["prompt_token_sha256"], "fixed-trace")
        row = summary["comparisons"][0]
        self.assertEqual(row["control"]["output_throughput"], 2.0)
        self.assertEqual(row["optimized"]["output_throughput"], 4.0)
        self.assertEqual(row["control"]["ttft_ms"]["p50"], 200.0)
        self.assertEqual(row["optimized"]["tpot_ms"]["p99"], 29.8)
        self.assertEqual(row["delta"]["output_throughput_percent"], 100.0)
        self.assertEqual(row["delta"]["ttft_p50_percent"], -50.0)
        self.assertEqual(row["delta"]["peak_allocated_bytes_percent"], 50.0)

    def test_rejects_input_trace_mismatch(self):
        summarize_ab_report = self._summarize()

        control = _report(fused=False, result={"backend": "marlin", "batch_size": 1, "repeat": 0, "requests": []})
        optimized = _report(fused=True, result={"backend": "marlin", "batch_size": 1, "repeat": 0, "requests": []})
        optimized["input"]["prompt_token_sha256"] = "different-trace"

        with self.assertRaisesRegex(ValueError, "prompt token trace"):
            summarize_ab_report(control=control, optimized=optimized)


if __name__ == "__main__":
    unittest.main()
