import json
import unittest
from pathlib import Path


class PDPhaseMapSuiteTests(unittest.TestCase):
    def test_smoke_suite_covers_four_phase_map_workloads_with_matched_variants(self):
        from benchmarks.suite import expand_suite

        suite_path = (
            Path(__file__).resolve().parents[1]
            / "benchmarks"
            / "suites"
            / "pd-phase-map-smoke.json"
        )
        suite = json.loads(suite_path.read_text())
        points = expand_suite(suite)

        self.assertEqual(suite["runs"], 1)
        self.assertEqual(set(suite["profiles"]), {"w0-short", "w2-decode", "w4-token-balanced", "w5-prefill-stress"})
        self.assertEqual(len(points), 24)
        self.assertEqual({point["request_rate"] for point in points}, {2.0, 4.0, 6.0})
        self.assertTrue(all(point["warmup_seconds"] == 10 for point in points))
        self.assertTrue(all(point["measurement_seconds"] == 20 for point in points))
        self.assertEqual(
            {point["variant"] for point in points},
            {"dual-collocated", "pd-shared-1p1d"},
        )

        grouped = {}
        for point in points:
            key = (point["experiment"], point["request_rate"], point["run"])
            grouped.setdefault(key, []).append(point)
        self.assertTrue(all(len(group) == 2 for group in grouped.values()))
        for group in grouped.values():
            self.assertEqual({point["workload_seed"] for point in group}, {0})
            self.assertEqual({point["arrival_seed"] for point in group}, {0})

        token_balanced = suite["profiles"]["w4-token-balanced"]
        self.assertEqual(token_balanced["trace_order"], "replica_balanced")
        self.assertEqual(token_balanced["classes"], [
            {"name": "short", "weight": 94, "input_len": 128, "output_len": 64},
            {"name": "long", "weight": 6, "input_len": 2048, "output_len": 64},
        ])


if __name__ == "__main__":
    unittest.main()
