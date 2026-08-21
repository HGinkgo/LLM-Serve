import unittest


class MoeGptqGemmBenchmarkTest(unittest.TestCase):

    def test_parse_m_values_deduplicates_and_requires_positive_integers(self):
        from benchmarks.moe_gptq_gemm import parse_m_values

        self.assertEqual(parse_m_values("1,4,8,4,32"), (1, 4, 8, 32))
        with self.assertRaisesRegex(ValueError, "positive integers"):
            parse_m_values("1,0,8")

    def test_summarize_timings_reports_percentiles_and_effective_tflops(self):
        from benchmarks.moe_gptq_gemm import summarize_timings

        result = summarize_timings(
            [1.0, 2.0, 3.0],
            m=4,
            k=128,
            n=256,
        )

        self.assertEqual(result["count"], 3)
        self.assertEqual(result["mean_ms"], 2.0)
        self.assertEqual(result["p50_ms"], 2.0)
        self.assertEqual(result["p99_ms"], 2.98)
        self.assertEqual(result["min_ms"], 1.0)
        self.assertEqual(result["max_ms"], 3.0)
        self.assertAlmostEqual(
            result["effective_tflops"],
            (2 * 4 * 128 * 256) / (0.002 * 1e12),
        )

    def test_build_result_row_keeps_workload_shape_explicit(self):
        from benchmarks.moe_gptq_gemm import build_result_row

        row = build_result_row(
            backend="marlin",
            m=8,
            k=128,
            n=256,
            group_size=128,
            timings_ms=[2.0],
        )

        self.assertEqual(
            row["shape"],
            {"m": 8, "k": 128, "n": 256, "group_size": 128},
        )
        self.assertEqual(row["backend"], "marlin")
        self.assertEqual(row["timing_ms"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
