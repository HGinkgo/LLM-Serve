import json
import tempfile
import unittest
from pathlib import Path


def _latency(value):
    return {
        "ttft": {"p50": value, "p99": value * 2},
        "tpot": {"p50": value / 2, "p99": value},
        "e2e": {"p50": value * 4, "p99": value * 5},
    }


def _result(experiment, variant, run):
    runtime = {
        "enable_chunked_prefill": True,
        "enable_kv_capacity_admission": True,
        "max_num_batched_tokens": 1024,
        "max_num_seqs": 128,
        "max_model_len": 2304 if experiment.endswith("mixed") else 512,
        "gpu_memory_utilization": 0.9,
        "random_seed": 20260816,
    }
    if variant == "dual-collocated":
        runtime.update({"dual_collocated": True, "collocated_gpus": [0, 1]})
    if variant == "pd-shared":
        runtime.update({
            "pd": True,
            "prefill_gpu": 0,
            "decode_gpu": 1,
            "kv_slot_count": 2,
            "kv_slot_capacity_tokens": 8192,
            "enable_pd_transport_overlap": True,
        })
    long_short = experiment.endswith("mixed")
    classes = [
        {"name": "short", "weight": 1, "input_len": 128, "output_len": 64},
    ]
    if long_short:
        classes.append(
            {"name": "long", "weight": 1, "input_len": 2048, "output_len": 64}
        )
    scale = {"strong-collocated": 1.0, "dual-collocated": 1.5, "pd-shared": 1.3}[variant]
    metrics = {
        "throughput": {
            "requests_per_second": 10 * scale,
            "output_tokens_per_second": 640 * scale,
        },
        "latency": {"overall": _latency(0.1 / scale), "short": _latency(0.1 / scale)},
        "cuda_graph": {"enabled": True, "captured_graphs": 12, "replays": 100},
        "pd": {},
        "dual_collocated": {},
    }
    if long_short:
        metrics["latency"]["long"] = _latency(0.4 / scale)
    if variant == "dual-collocated":
        metrics["dual_collocated"] = {
            "policy": "two_request_striped_round_robin",
            "assigned_requests": {"replica-0": 20, "replica-1": 20},
        }
        metrics["cuda_graph"]["replicas"] = {
            worker_id: {"enabled": True, "captured_graphs": 6, "replays": 50}
            for worker_id in ("replica-0", "replica-1")
        }
    if variant == "pd-shared":
        metrics["pd"] = {
            "prefill_batches_detail": [{
                "request_ids": [0, 1],
                "batch_size": 2,
                "transports": ["shared_slot", "shared_slot"],
            }],
            "slot_release_samples": [{"slot_stats": {
                "free_slots": 2, "ready_slots": 0,
                "consuming_slots": 0, "pending_transfers": 0,
            }}],
            "queue_samples": [{
                "pending_prefill_requests": 3,
                "active_decode_requests": 5,
            }],
            "decode_idle": {"counts": {"waiting_prefill_output": 2}, "duration_ms": {"waiting_prefill_output": 4.0}},
        }
    request_trace = {
        "generator": "iter_request_specs",
        "ordering": "interleaved",
        "entry_count": 2,
        "sha256": f"run-{run}",
        "entries": [
            {"request_id": 0, "request_class": "short", "input_len": 128, "output_len": 64, "prompt_sha256": "a"},
            {"request_id": 1, "request_class": "long" if long_short else "short", "input_len": 2048 if long_short else 128, "output_len": 64, "prompt_sha256": "b"},
        ],
    }
    return {
        "schema_version": 2,
        "complete": True,
        "point_id": f"{experiment}-{variant}-concurrency-64-r{run + 1}",
        "git_commit": "abc123",
        "config": {
            "experiment": experiment,
            "variant": variant,
            "run": run,
            "workload_seed": run,
            "model": "Qwen3-8B",
            "model_revision": "test-revision",
            "sampling": {
                "temperature": 0.01,
                "ignore_eos": True,
                "max_tokens": "per_request_output_len",
            },
            "runtime": runtime,
            "workload": {"trace_order": "interleaved", "classes": classes},
        },
        "metrics": metrics,
        "requests": [
            {"request_id": 0, "request_class": "short", "timeline": {"t_submit": 1.0, "t_decode_admitted": 1.2, "t_finish": 1.5}},
            {"request_id": 1, "request_class": "long" if long_short else "short", "timeline": {"t_submit": 1.0, "t_prefill_first_scheduled": 1.1, "t_prefill_finish": 1.3, "t_finish": 1.6}},
        ],
        "telemetry": {
            "measurement_window": {"start": 0.0, "end": 60.0},
            "request_trace": request_trace,
            "effective_runtime": {"kind": "dual_collocated" if variant == "dual-collocated" else "pd" if variant == "pd-shared" else "collocated"},
            "scheduler_steps": [{
                "replica_queue_state": {
                    "replica-0": {"running_queue_size": 5, "decode_request_count": 5},
                    "replica-1": {"running_queue_size": 5, "decode_request_count": 5},
                }
            }],
        },
    }


class ResourceEquivalentReportTests(unittest.TestCase):
    def test_report_compares_equal_resource_deployments_and_mixed_overlap(self):
        from benchmarks.pd_resource_equivalent import build_report

        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            runs = results_dir / "runs"
            runs.mkdir()
            (results_dir / "manifest.json").write_text(json.dumps({
                "complete": True,
                "git_commit": "abc123",
                "execution_order": [],
            }))
            for experiment in (
                "pd-resource-uniform-short",
                "pd-resource-long-short-mixed",
            ):
                for variant in ("strong-collocated", "dual-collocated", "pd-shared"):
                    for run in range(3):
                        result = _result(experiment, variant, run)
                        (runs / f"{result['point_id']}.json").write_text(json.dumps(result))

            report_path = build_report(results_dir)
            report = report_path.read_text()

            self.assertIn("B vs A", report)
            self.assertIn("C vs B", report)
            self.assertIn("C vs A", report)
            self.assertIn("Long Prefill / short Decode overlap", report)
            self.assertIn("striped round-robin", report)
            self.assertIn("shared_slot", report)
            self.assertIn("pure PD", report)
            self.assertIn("Chunked Prefill Evidence", report)
            self.assertIn("Prefill queue wait", report)
            self.assertTrue((results_dir / "resource_equivalent_summary.csv").exists())
            self.assertTrue((results_dir / "trace_audit.json").exists())

    def test_report_rejects_nonshared_pd_transport(self):
        from benchmarks.pd_resource_equivalent import build_report

        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            runs = results_dir / "runs"
            runs.mkdir()
            (results_dir / "manifest.json").write_text(json.dumps({}))
            for variant in ("strong-collocated", "dual-collocated", "pd-shared"):
                result = _result("pd-resource-uniform-short", variant, 0)
                if variant == "pd-shared":
                    result["metrics"]["pd"]["prefill_batches_detail"][0]["transports"] = ["inline"]
                (runs / f"{variant}.json").write_text(json.dumps(result))

            with self.assertRaisesRegex(ValueError, "shared_slot"):
                build_report(results_dir)

    def test_report_rejects_results_without_cuda_graph_replay(self):
        from benchmarks.pd_resource_equivalent import build_report

        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            runs = results_dir / "runs"
            runs.mkdir()
            for variant in ("strong-collocated", "dual-collocated", "pd-shared"):
                result = _result("pd-resource-uniform-short", variant, 0)
                if variant == "dual-collocated":
                    result["metrics"]["cuda_graph"]["replays"] = 0
                (runs / f"{variant}.json").write_text(json.dumps(result))

            with self.assertRaisesRegex(ValueError, "CUDA Graph"):
                build_report(results_dir)

    def test_replica_rows_keep_per_class_assignment(self):
        from benchmarks.pd_resource_equivalent import _replica_rows

        result = _result("pd-resource-long-short-mixed", "dual-collocated", 0)
        result["requests"] = [
            {"replica_id": "replica-0", "request_class": "short"},
            {"replica_id": "replica-0", "request_class": "long"},
            {"replica_id": "replica-1", "request_class": "short"},
            {"replica_id": "replica-1", "request_class": "long"},
        ]

        rows = {row["replica"]: row for row in _replica_rows([result])}

        self.assertEqual(rows["replica-0"]["short_requests"], 1)
        self.assertEqual(rows["replica-0"]["long_requests"], 1)
        self.assertEqual(rows["replica-1"]["short_requests"], 1)
        self.assertEqual(rows["replica-1"]["long_requests"], 1)

    def test_report_rejects_a_noninterleaved_mixed_trace(self):
        from benchmarks.pd_resource_equivalent import build_report

        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            runs = results_dir / "runs"
            runs.mkdir()
            for variant in ("strong-collocated", "dual-collocated", "pd-shared"):
                result = _result("pd-resource-long-short-mixed", variant, 0)
                result["telemetry"]["request_trace"]["entries"][1]["request_class"] = "short"
                (runs / f"{variant}.json").write_text(json.dumps(result))

            with self.assertRaisesRegex(ValueError, "interleaved trace"):
                build_report(results_dir)


if __name__ == "__main__":
    unittest.main()
