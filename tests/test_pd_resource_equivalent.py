import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ResourceEquivalentSuiteTests(unittest.TestCase):
    def test_formal_suite_covers_uniform_and_interleaved_mixed_workloads(self):
        from benchmarks.suite import expand_suite

        suite = json.loads(
            (ROOT / "benchmarks/suites/pd-resource-equivalent-formal.json").read_text()
        )
        points = expand_suite(suite)

        self.assertEqual(suite["runs"], 3)
        self.assertEqual(len(points), 18)
        self.assertEqual({point["max_concurrency"] for point in points}, {64})
        self.assertEqual(
            {point["variant"] for point in points},
            {"strong-collocated", "dual-collocated", "pd-shared"},
        )
        self.assertEqual(
            {point["experiment"] for point in points},
            {"pd-resource-uniform-short", "pd-resource-long-short-mixed"},
        )
        self.assertTrue(all(
            point["warmup_seconds"] == 30
            and point["measurement_seconds"] == 60
            and point["runtime"]["enable_chunked_prefill"]
            and point["runtime"]["enable_kv_capacity_admission"]
            and point["runtime"]["max_num_batched_tokens"] == 1024
            and point["runtime"]["max_num_seqs"] == 128
            for point in points
        ))

        mixed_points = [
            point for point in points
            if point["experiment"] == "pd-resource-long-short-mixed"
        ]
        self.assertTrue(all(
            point["workload"]["trace_order"] == "interleaved"
            and point["workload"]["classes"] == [
                {"name": "short", "weight": 1, "input_len": 128, "output_len": 64},
                {"name": "long", "weight": 1, "input_len": 2048, "output_len": 64},
            ]
            and point["runtime"]["max_model_len"] == 2304
            for point in mixed_points
        ))

    def test_dual_collocated_and_pd_shared_use_distinct_nonhybrid_paths(self):
        from benchmarks.suite import expand_suite

        suite = json.loads(
            (ROOT / "benchmarks/suites/pd-resource-equivalent-formal.json").read_text()
        )
        points = expand_suite(suite)

        for experiment in {point["experiment"] for point in points}:
            run_zero = [
                point for point in points
                if point["experiment"] == experiment and point["run"] == 0
            ]
            by_variant = {point["variant"]: point for point in run_zero}
            baseline = by_variant["strong-collocated"]["runtime"]
            dual = by_variant["dual-collocated"]["runtime"]
            shared = by_variant["pd-shared"]["runtime"]

            self.assertFalse(baseline.get("pd", False))
            self.assertFalse(baseline.get("dual_collocated", False))
            self.assertTrue(dual["dual_collocated"])
            self.assertFalse(dual.get("pd", False))
            self.assertEqual(dual["collocated_gpus"], [0, 1])
            self.assertEqual(len(dual["collocated_init_methods"]), 2)
            self.assertTrue(shared["pd"])
            self.assertFalse(shared.get("dual_collocated", False))
            self.assertEqual(shared["prefill_gpu"], 0)
            self.assertEqual(shared["decode_gpu"], 1)
            self.assertEqual(shared["kv_slot_count"], 2)
            self.assertTrue(shared["enable_pd_transport_overlap"])
            self.assertFalse(shared["decode_enforce_eager"])
            for field in (
                "max_model_len",
                "max_num_batched_tokens",
                "max_num_seqs",
                "gpu_memory_utilization",
                "enable_chunked_prefill",
                "enable_kv_capacity_admission",
                "random_seed",
            ):
                self.assertEqual(baseline[field], dual[field])
                self.assertEqual(baseline[field], shared[field])

        mixed_shared = next(
            point for point in points
            if point["experiment"] == "pd-resource-long-short-mixed"
            and point["variant"] == "pd-shared"
        )
        self.assertEqual(mixed_shared["runtime"]["kv_slot_capacity_tokens"], 8192)


if __name__ == "__main__":
    unittest.main()
