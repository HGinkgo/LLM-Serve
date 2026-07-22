import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SUITE = {
    "schema_version": 1,
    "name": "serving-v1",
    "runs": 2,
    "profiles": {
        "mixed": {
            "classes": [
                {"name": "short", "weight": 0.8, "input_len": 128, "output_len": 64},
                {"name": "long", "weight": 0.2, "input_len": 4096, "output_len": 64},
            ]
        }
    },
    "experiments": [
        {
            "name": "chunked-poisson",
            "arrival": "poisson",
            "profile": "mixed",
            "num_requests": 20,
            "request_rates": [1.0, 2.0],
            "variants": [
                {"name": "baseline"},
                {"name": "chunked", "enable_chunked_prefill": True},
            ],
            "runtime": {"max_model_len": 4608, "max_num_batched_tokens": 512},
            "slo_ms": {"ttft": 2000, "tpot": 100},
        }
    ],
}

CLOSED_LOOP_SUITE = {
    "schema_version": 1,
    "name": "serving-v1",
    "runs": 2,
    "profiles": SUITE["profiles"],
    "experiments": [
        {
            "name": "decode-closed-loop",
            "arrival": "closed-loop",
            "profile": "mixed",
            "max_concurrencies": [1, 4],
            "warmup_seconds": 2,
            "measurement_seconds": 5,
            "variants": [
                {"name": "baseline"},
                {"name": "eagle", "enable_speculative": True},
            ],
            "runtime": {"max_model_len": 4608, "max_num_batched_tokens": 512},
        }
    ],
}


class BenchmarkSuiteTests(unittest.TestCase):
    def test_repository_suite_configs_expand_to_unique_points(self):
        from benchmarks.suite import expand_suite

        suite_dir = Path(__file__).resolve().parents[1] / "benchmarks" / "suites"
        suite_paths = sorted(suite_dir.glob("*.json"))

        self.assertEqual(
            {path.name for path in suite_paths},
            {"pilot.json", "smoke.json"},
        )
        for path in suite_paths:
            points = expand_suite(json.loads(path.read_text()))
            self.assertTrue(points)
            self.assertEqual(
                len({point["point_id"] for point in points}), len(points)
            )

    def test_expand_suite_pairs_variants_with_identical_randomness(self):
        self.assertIsNotNone(importlib.util.find_spec("benchmarks.suite"))
        from benchmarks.suite import expand_suite

        points = expand_suite(SUITE)

        self.assertEqual(len(points), 8)
        self.assertEqual(len({point["point_id"] for point in points}), 8)
        paired = {}
        for point in points:
            key = (point["request_rate"], point["run"])
            paired.setdefault(key, []).append(point)
        for variants in paired.values():
            self.assertEqual({point["workload_seed"] for point in variants}, {variants[0]["run"]})
            self.assertEqual({point["arrival_seed"] for point in variants}, {variants[0]["run"]})
            self.assertEqual(
                {point["variant"] for point in variants},
                {"baseline", "chunked"},
            )
        chunked = next(point for point in points if point["variant"] == "chunked")
        self.assertTrue(chunked["runtime"]["enable_chunked_prefill"])
        self.assertEqual(chunked["workload"]["classes"][1]["input_len"], 4096)

    def test_expand_suite_rejects_unknown_profile(self):
        self.assertIsNotNone(importlib.util.find_spec("benchmarks.suite"))
        from benchmarks.suite import expand_suite

        invalid = json.loads(json.dumps(SUITE))
        invalid["experiments"][0]["profile"] = "missing"

        with self.assertRaisesRegex(ValueError, "unknown profile"):
            expand_suite(invalid)

    def test_expand_suite_supports_closed_loop_concurrency_sweep(self):
        from benchmarks.suite import expand_suite

        points = expand_suite(CLOSED_LOOP_SUITE)

        self.assertEqual(len(points), 8)
        self.assertEqual({point["max_concurrency"] for point in points}, {1, 4})
        self.assertTrue(all(point["warmup_seconds"] == 2 for point in points))
        self.assertTrue(all(point["measurement_seconds"] == 5 for point in points))
        paired = {}
        for point in points:
            key = (point["max_concurrency"], point["run"])
            paired.setdefault(key, []).append(point)
        for variants in paired.values():
            self.assertEqual({point["variant"] for point in variants}, {"baseline", "eagle"})
            self.assertEqual(len({point["workload_seed"] for point in variants}), 1)

    def test_resume_requires_complete_matching_result(self):
        self.assertIsNotNone(importlib.util.find_spec("benchmarks.suite"))
        from benchmarks.suite import can_resume_result, expand_suite

        point = expand_suite(SUITE)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps({
                "schema_version": 2,
                "complete": True,
                "point_id": point["point_id"],
                "git_commit": "abc123",
            }))

            self.assertTrue(can_resume_result(path, point, "abc123"))
            self.assertFalse(can_resume_result(path, point, "different"))
            path.write_text("not-json")
            self.assertFalse(can_resume_result(path, point, "abc123"))

    def test_aggregate_results_reports_sample_stddev_and_baseline_ratio(self):
        self.assertIsNotNone(importlib.util.find_spec("benchmarks.suite"))
        from benchmarks.suite import aggregate_results

        results = []
        for run, baseline, chunked in [(0, 10.0, 12.0), (1, 14.0, 18.0)]:
            for variant, throughput in [("baseline", baseline), ("chunked", chunked)]:
                results.append({
                    "complete": True,
                    "config": {
                        "experiment": "chunked-poisson",
                        "arrival": "poisson",
                        "variant": variant,
                        "request_rate": 1.0,
                        "run": run,
                    },
                    "metrics": {
                        "throughput": {"requests_per_second": throughput},
                    },
                })

        rows = aggregate_results(results, "metrics.throughput.requests_per_second")

        baseline = next(row for row in rows if row["variant"] == "baseline")
        chunked = next(row for row in rows if row["variant"] == "chunked")
        self.assertEqual(baseline["mean"], 12.0)
        self.assertAlmostEqual(baseline["stddev"], 2.8284271247461903)
        self.assertEqual(baseline["ratio_to_baseline"], 1.0)
        self.assertEqual(chunked["mean"], 15.0)
        self.assertEqual(chunked["ratio_to_baseline"], 1.25)

    def test_summary_rows_flatten_standard_metrics_in_public_units(self):
        from benchmarks.suite import build_summary_rows

        result = {
            "point_id": "point-1",
            "complete": True,
            "git_commit": "abc123",
            "config": {
                "experiment": "chunked-poisson",
                "arrival": "poisson",
                "variant": "baseline",
                "request_rate": 1.0,
                "run": 0,
            },
            "metrics": {
                "completed": 4,
                "failed": 0,
                "throughput": {
                    "requests_per_second": 2.0,
                    "output_tokens_per_second": 8.0,
                },
                "latency": {
                    "overall": {
                        "ttft": {"p50": 0.1, "p99": 0.5},
                        "tpot": {"p50": 0.02, "p99": 0.04},
                    }
                },
                "speculative": {"acceptance_rate": None},
            },
        }

        row = build_summary_rows([result])[0]

        self.assertEqual(row["point_id"], "point-1")
        self.assertEqual(row["request_throughput_rps"], 2.0)
        self.assertEqual(row["output_throughput_tps"], 8.0)
        self.assertEqual(row["ttft_p50_ms"], 100.0)
        self.assertEqual(row["tpot_p99_ms"], 40.0)
        self.assertIsNone(row["acceptance_rate"])


if __name__ == "__main__":
    unittest.main()
