import importlib.util
import unittest
from collections import Counter


class BenchmarkWorkloadTests(unittest.TestCase):
    def test_weighted_workload_is_exact_and_reproducible(self):
        self.assertIsNotNone(importlib.util.find_spec("benchmarks.workloads"))
        from benchmarks.workloads import WorkloadClass, build_request_specs

        classes = [
            WorkloadClass("short", weight=0.8, input_len=128, output_len=64),
            WorkloadClass("long", weight=0.2, input_len=4096, output_len=64),
        ]

        first = build_request_specs(classes, num_requests=10, seed=7)
        second = build_request_specs(classes, num_requests=10, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(
            Counter(spec.request_class for spec in first),
            {"short": 8, "long": 2},
        )
        self.assertEqual([spec.request_id for spec in first], list(range(10)))
        self.assertTrue(all(len(spec.prompt_token_ids) == spec.input_len for spec in first))

    def test_workload_seed_changes_generated_tokens(self):
        self.assertIsNotNone(importlib.util.find_spec("benchmarks.workloads"))
        from benchmarks.workloads import WorkloadClass, build_request_specs

        classes = [WorkloadClass("short", weight=1, input_len=8, output_len=4)]

        first = build_request_specs(classes, num_requests=2, seed=1)
        second = build_request_specs(classes, num_requests=2, seed=2)

        self.assertNotEqual(first[0].prompt_token_ids, second[0].prompt_token_ids)


class BenchmarkArrivalTests(unittest.TestCase):
    def test_poisson_arrivals_start_immediately_and_are_reproducible(self):
        self.assertIsNotNone(importlib.util.find_spec("benchmarks.arrivals"))
        from benchmarks.arrivals import poisson_arrival_times

        first = poisson_arrival_times(num_requests=5, request_rate=2.0, seed=11)
        second = poisson_arrival_times(num_requests=5, request_rate=2.0, seed=11)

        self.assertEqual(first, second)
        self.assertEqual(first[0], 0.0)
        self.assertTrue(all(left < right for left, right in zip(first, first[1:])))

    def test_poisson_arrivals_validate_inputs(self):
        self.assertIsNotNone(importlib.util.find_spec("benchmarks.arrivals"))
        from benchmarks.arrivals import poisson_arrival_times

        self.assertEqual(poisson_arrival_times(0, request_rate=1.0, seed=0), [])
        with self.assertRaisesRegex(ValueError, "num_requests"):
            poisson_arrival_times(-1, request_rate=1.0, seed=0)
        with self.assertRaisesRegex(ValueError, "request_rate"):
            poisson_arrival_times(1, request_rate=0.0, seed=0)


class BenchmarkMetricTests(unittest.TestCase):
    def test_serving_summary_matches_hand_calculated_values(self):
        self.assertIsNotNone(importlib.util.find_spec("benchmarks.metrics"))
        from benchmarks.metrics import summarize_serving_run

        requests = [
            {
                "request_class": "short",
                "prompt_tokens": 10,
                "output_tokens": 3,
                "success": True,
                "arrival_time": 0.0,
                "first_token_time": 1.0,
                "token_times": [1.0, 2.0, 3.0],
                "finish_time": 3.0,
            },
            {
                "request_class": "long",
                "prompt_tokens": 20,
                "output_tokens": 2,
                "success": True,
                "arrival_time": 0.0,
                "first_token_time": 2.0,
                "token_times": [2.0, 4.0],
                "finish_time": 4.0,
            },
        ]

        result = summarize_serving_run(
            requests,
            duration=4.0,
            slo_ms={"ttft": 1500, "tpot": 1500, "e2e": 3500},
        )

        self.assertEqual(result["completed"], 2)
        self.assertEqual(result["failed"], 0)
        self.assertAlmostEqual(result["throughput"]["requests_per_second"], 0.5)
        self.assertAlmostEqual(result["throughput"]["input_tokens_per_second"], 7.5)
        self.assertAlmostEqual(result["throughput"]["output_tokens_per_second"], 1.25)
        self.assertAlmostEqual(result["throughput"]["total_tokens_per_second"], 8.75)
        self.assertAlmostEqual(result["latency"]["overall"]["ttft"]["p50"], 1.5)
        self.assertAlmostEqual(result["latency"]["overall"]["ttft"]["p90"], 1.9)
        self.assertAlmostEqual(result["latency"]["overall"]["tpot"]["p50"], 1.5)
        self.assertAlmostEqual(result["latency"]["overall"]["e2e"]["p99"], 3.99)
        self.assertEqual(result["latency"]["short"]["ttft"]["p50"], 1.0)
        self.assertEqual(result["latency"]["long"]["ttft"]["p50"], 2.0)
        self.assertAlmostEqual(result["goodput"]["requests_per_second"], 0.25)
        self.assertEqual(result["goodput"]["completed"], 1)

    def test_serving_summary_keeps_failures_out_of_latency_and_token_totals(self):
        self.assertIsNotNone(importlib.util.find_spec("benchmarks.metrics"))
        from benchmarks.metrics import summarize_serving_run

        requests = [
            {
                "request_class": "short",
                "prompt_tokens": 10,
                "output_tokens": 0,
                "success": False,
                "arrival_time": 0.0,
                "first_token_time": None,
                "token_times": [],
                "finish_time": None,
            }
        ]

        result = summarize_serving_run(requests, duration=2.0)

        self.assertEqual(result["completed"], 0)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["throughput"]["total_tokens_per_second"], 0.0)
        self.assertIsNone(result["latency"]["overall"]["ttft"]["p50"])
        self.assertIsNone(result["goodput"])


if __name__ == "__main__":
    unittest.main()
