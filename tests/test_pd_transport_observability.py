import csv
import json
import tempfile
import unittest
from pathlib import Path


def _result(run: int):
    return {
        "complete": True,
        "point_id": f"pd-transport-observability-pd-shared-c64-r{run}",
        "config": {
            "variant": "pd-shared",
            "run": run - 1,
            "runtime": {
                "pd": True,
                "prefill_gpu": 0,
                "decode_gpu": 1,
                "kv_slot_count": 2,
            },
        },
        "requests": [{
            "request_id": run,
            "success": True,
            "timeline": {
                "t_submit": 1.0,
                "t_prefill_first_scheduled": 1.1,
                "t_prefill_finish": 1.3,
                "t_first_token": 1.3,
                "t_slot_acquired": 1.05,
                "t_kv_export_started": 1.31,
                "t_kv_export_finished": 1.40,
                "t_slot_ready": 1.42,
                "t_decode_descriptor_received": 1.46,
                "t_slot_consuming": 1.465,
                "t_decode_admission_started": 1.47,
                "t_decode_admitted": 1.48,
                "t_handoff_finish": 1.50,
                "t_decode_admission_returned": 1.51,
                "t_finish": 2.0,
            },
        }],
        "metrics": {
            "pd": {
                "prefill_batches_detail": [{
                    "transports": ["shared_slot"],
                    "slot_observability_after_ready": {
                        "environment": {
                            "cpu_affinity": [0, 1],
                            "shared_memory_numa": {"page_counts": {"0": 8}},
                        },
                    },
                }],
                "slot_release_samples": [{
                    "transfer_ids": [f"transfer-{run}"],
                    "slot_stats": {
                        "free_slots": 2,
                        "ready_slots": 0,
                        "consuming_slots": 0,
                        "pending_transfers": 0,
                        "observability": {
                            "events": [
                                {"slot_id": 0, "generation": run, "state": "filling", "at": 1.05, "writer": "prefill_worker.slot_pool"},
                                {"slot_id": 0, "generation": run, "state": "ready", "at": 1.42, "writer": "prefill_worker.slot_pool"},
                                {"slot_id": 0, "generation": run, "state": "consuming", "at": 1.48, "writer": "prefill_worker.ack"},
                                {"slot_id": 0, "generation": run, "state": "free", "at": 1.51, "writer": "prefill_worker.ack"},
                            ],
                        },
                    },
                }],
                "decode_idle": {
                    "counts": {
                        "no_runnable_request": 0,
                        "waiting_prefill_output": run,
                        "waiting_kv_h2d": 0,
                        "scheduler_not_scheduled": 0,
                    },
                    "duration_ms": {
                        "no_runnable_request": 0.0,
                        "waiting_prefill_output": float(run),
                        "waiting_kv_h2d": 0.0,
                        "scheduler_not_scheduled": 0.0,
                    },
                },
                "worker_health": {
                    "prefill": {"environment": {"cpu_affinity": [0, 1]}},
                    "decode": {"environment": {"cpu_affinity": [2, 3]}},
                },
            },
        },
        "telemetry": {"measurement_window": {"start": 0.0, "end": 3.0}},
    }


class PDTransportObservabilityReportTests(unittest.TestCase):

    def test_observability_suite_keeps_the_current_pd_shared_configuration(self):
        from benchmarks.suite import expand_suite

        root = Path(__file__).resolve().parents[1]
        suite = json.loads(
            (root / "benchmarks/suites/pd-transport-observability.json").read_text()
        )
        points = expand_suite(suite)

        self.assertEqual(suite["runs"], 3)
        self.assertEqual(len(points), 3)
        self.assertEqual({point["variant"] for point in points}, {
            "pd-shared-observability",
        })
        for point in points:
            runtime = point["runtime"]
            self.assertTrue(runtime["pd"])
            self.assertEqual(point["max_concurrency"], 64)
            self.assertEqual(point["warmup_seconds"], 30)
            self.assertEqual(point["measurement_seconds"], 60)
            self.assertEqual(runtime["prefill_batch_size"], 4)
            self.assertEqual(runtime["kv_slot_count"], 2)
            self.assertTrue(runtime["enable_latency_telemetry"])
            self.assertTrue(runtime["enable_chunked_prefill"])
            self.assertTrue(runtime["enable_kv_capacity_admission"])
            self.assertFalse(runtime["decode_enforce_eager"])

    def test_report_accepts_a_pd_shared_variant_name_from_the_observation_suite(self):
        from benchmarks.pd_transport_observability import build_report

        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            runs = results_dir / "runs"
            runs.mkdir()
            (results_dir / "manifest.json").write_text(json.dumps({}))
            for run in range(1, 4):
                result = _result(run)
                result["config"]["variant"] = "pd-shared-observability"
                (runs / f"run-{run}.json").write_text(json.dumps(result))

            self.assertTrue(build_report(results_dir).exists())

    def test_report_writes_auditable_timelines_idle_reasons_and_slot_lifecycle(self):
        from benchmarks.pd_transport_observability import build_report

        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            runs = results_dir / "runs"
            runs.mkdir()
            (results_dir / "manifest.json").write_text(json.dumps({
                "complete": True,
                "git_commit": "abc123",
                "completed_points": 3,
                "total_points": 3,
            }))
            for run in range(1, 4):
                (runs / f"run-{run}.json").write_text(json.dumps(_result(run)))

            report = build_report(results_dir).read_text()

            self.assertIn("H2D is enqueued, not completed", report)
            self.assertIn("waiting_prefill_output", report)
            self.assertIn("Shared-slot lifecycle", report)
            timeline_path = results_dir / "request_transport_timeline.csv"
            idle_path = results_dir / "decode_idle_summary.csv"
            slots_path = results_dir / "slot_lifecycle.csv"
            workers_path = results_dir / "worker_environment.csv"
            self.assertTrue(timeline_path.exists())
            self.assertTrue(idle_path.exists())
            self.assertTrue(slots_path.exists())
            self.assertTrue(workers_path.exists())
            with timeline_path.open() as input_file:
                timeline_rows = list(csv.DictReader(input_file))
            self.assertEqual(len(timeline_rows), 3)
            self.assertAlmostEqual(
                float(timeline_rows[0]["descriptor_queue_ms"]), 40.0
            )
            with slots_path.open() as input_file:
                slot_rows = list(csv.DictReader(input_file))
            self.assertEqual({row["state"] for row in slot_rows}, {
                "filling", "ready", "consuming", "free",
            })
            with workers_path.open() as input_file:
                worker_rows = list(csv.DictReader(input_file))
            self.assertEqual({row["role"] for row in worker_rows}, {
                "prefill", "decode",
            })

    def test_report_distinguishes_cuda_event_duration_from_host_observation(self):
        from benchmarks.pd_transport_observability import build_report

        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            runs = results_dir / "runs"
            runs.mkdir()
            (results_dir / "manifest.json").write_text(json.dumps({
                "complete": True,
                "git_commit": "abc123",
                "completed_points": 1,
                "total_points": 1,
            }))
            result = _result(1)
            timeline = result["requests"][0]["timeline"]
            timeline.update({
                "t_kv_import_enqueued": 1.50,
                "t_kv_import_completion_observed": 1.60,
                "kv_import_gpu_ms": 2.5,
                "copy_gpu_ms": 2.5,
                "decode_gpu_step_ms": 10.0,
                "copy_compute_overlap_ms": 2.0,
                "copy_compute_overlap_ratio": 0.8,
                "serial_gpu_ms": 12.5,
                "overlapped_makespan_gpu_ms": 10.5,
                "critical_path_reduction_gpu_ms": 2.0,
            })
            (runs / "run-1.json").write_text(json.dumps(result))

            report = build_report(results_dir).read_text()

            self.assertIn("target-side CUDA Event", report)
            self.assertIn("device-side copy/GPU-step overlap", report)
            self.assertIn("Pair coverage: `1/1`", report)
            with (results_dir / "request_transport_timeline.csv").open() as input_file:
                row = next(csv.DictReader(input_file))
            self.assertEqual(float(row["kv_import_gpu_ms"]), 2.5)
            self.assertAlmostEqual(
                float(row["kv_import_completion_observed_after_enqueue_ms"]),
                100.0,
            )
            self.assertAlmostEqual(
                float(row["copy_compute_overlap_ms"]),
                2.0,
            )

    def test_report_discloses_overlap_pair_coverage_per_variant(self):
        from benchmarks.pd_transport_observability import build_report

        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            runs = results_dir / "runs"
            runs.mkdir()
            (results_dir / "manifest.json").write_text(json.dumps({
                "complete": True,
                "git_commit": "abc123",
                "completed_points": 2,
                "total_points": 2,
            }))
            serial = _result(1)
            serial["config"]["variant"] = "serial"
            overlap = _result(2)
            overlap["config"]["variant"] = "overlap"
            overlap["requests"][0]["timeline"].update({
                "copy_compute_overlap_ms": 1.0,
                "decode_gpu_step_ms": 4.0,
            })
            (runs / "serial.json").write_text(json.dumps(serial))
            (runs / "overlap.json").write_text(json.dumps(overlap))

            report = build_report(results_dir).read_text()

            self.assertIn("| serial | 0/1 |", report)
            self.assertIn("| overlap | 1/1 |", report)


if __name__ == "__main__":
    unittest.main()
