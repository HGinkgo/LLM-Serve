import csv
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _result(variant, run, *, pd=False):
    base = 10.0 + run
    runtime = {
        "enable_chunked_prefill": True,
        "enable_kv_capacity_admission": True,
        "max_model_len": 512,
        "max_num_batched_tokens": 1024,
        "max_num_seqs": 128,
        "gpu_memory_utilization": 0.9,
        "enforce_eager": not pd,
        "enable_latency_telemetry": True,
        "sample_gpu_clocks": True,
    }
    if pd:
        runtime.update({
            "pd": True,
            "prefill_gpu": 0,
            "decode_gpu": 1,
            "prefill_batch_size": 4,
            "prefill_enforce_eager": True,
            "decode_enforce_eager": False,
            "kv_slot_count": 2,
            "kv_slot_capacity_tokens": 1024,
        })
    timeline = {
        "t_submit": base,
        "t_prefill_first_scheduled": base + 0.001,
        "t_prefill_finish": base + 0.011,
        "t_first_token": base + 0.011,
        "t_handoff_finish": base + (0.017 if pd else 0.011),
        "t_decode_admitted": base + (0.012 if pd else 0.011),
        "t_finish": base + 0.050,
        "writers": {
            "t_submit": "benchmark_harness",
            "t_first_token": "prefill_worker.model_runner" if pd else "collocated_model_runner",
        },
    }
    pd_metrics = {}
    if pd:
        pd_metrics = {
            "prefill_batches_detail": [{
                "request_ids": [run],
                "batch_size": 1,
                "handoff_path_ms": 6.0,
                "shared_slot_id": 0,
                "shared_slot_generation": run + 1,
                "transport": "shared_slot",
            }],
            "slot_release_samples": [{
                "transfer_ids": [f"transfer-{run}"],
                "slot_stats": {
                    "free_slots": 2,
                    "ready_slots": 0,
                    "consuming_slots": 0,
                    "pending_transfers": 0,
                },
            }],
            "worker_health": {
                "prefill": {"alive": True, "exitcode": None},
                "decode": {"alive": True, "exitcode": None},
            },
            "fatal_error": None,
        }
    return {
        "schema_version": 2,
        "complete": True,
        "point_id": f"pd-strong-{variant}-concurrency-64-r{run + 1}",
        "git_commit": "abc123",
        "metadata": {
            "git_commit": "abc123",
            "gpus": [{"index": 0, "name": "GPU0"}, {"index": 1, "name": "GPU1"}],
            "cuda_peer_access": {"0->1": True, "1->0": True},
            "nvidia_smi_topology": "GPU0 GPU1 PHB",
        },
        "config": {
            "variant": variant,
            "run": run,
            "runtime": runtime,
            "workload_seed": run,
            "workload": {
                "classes": [{
                    "name": "decode",
                    "weight": 1,
                    "input_len": 128,
                    "output_len": 64,
                }],
            },
        },
        "metrics": {
            "throughput": {
                "requests_per_second": 4.0,
                "output_tokens_per_second": 256.0,
            },
            "cuda_graph": {
                "enabled": True,
                "captured_graphs": 12,
                "replays": 100 + run,
                "fallbacks": {"prefill": 1},
            },
            "pd": pd_metrics,
        },
        "requests": [{
            "request_id": run,
            "request_class": "decode",
            "success": True,
            "ttft_ms": 11.0,
            "tpot_ms": 1.0,
            "e2e_ms": 50.0,
            "timeline": timeline,
        }],
        "telemetry": {
            "measurement_window": {"start": base - 1, "end": base + 1},
            "effective_runtime": {"kind": "pd" if pd else "collocated"},
            "gpu_clocks": {"0": {"sm_mhz": {"min": 1000, "max": 1800}}},
            "scheduler_steps": [{
                "waiting_queue_size": 1,
                "running_queue_size": 2,
                "prefill_token_count": 128,
                "decode_token_count": 63,
                "prefill_request_count": 1,
                "decode_request_count": 1,
                "remaining_token_budget": 833,
                "chunked_prefill": True,
                "partial_prefill_chunk_count": 0,
            }],
        },
    }


class StrongBaselineSuiteTests(unittest.TestCase):
    def test_primary_suite_is_only_strong_collocated_and_shared_pd(self):
        from benchmarks.suite import expand_suite

        suite = json.loads(
            (ROOT / "benchmarks/suites/pd-strong-baseline.json").read_text()
        )
        points = expand_suite(suite)

        self.assertEqual(suite["runs"], 3)
        self.assertEqual(len(points), 6)
        self.assertEqual({point["variant"] for point in points}, {
            "strong-collocated", "pd-shared",
        })
        self.assertEqual({point["max_concurrency"] for point in points}, {64})
        for point in points:
            runtime = point["runtime"]
            self.assertTrue(runtime["enable_chunked_prefill"])
            self.assertTrue(runtime["enable_kv_capacity_admission"])
            self.assertEqual(runtime["max_num_batched_tokens"], 1024)
            self.assertEqual(runtime["max_num_seqs"], 128)
            self.assertEqual(point["warmup_seconds"], 30)
            self.assertEqual(point["measurement_seconds"], 60)
            self.assertTrue(runtime["enable_latency_telemetry"])
            self.assertTrue(runtime["sample_gpu_clocks"])
            self.assertFalse(runtime.get("enable_speculative", False))
            self.assertNotIn("awq_backend", runtime)
        collocated = next(point for point in points if point["variant"] == "strong-collocated")
        self.assertFalse(collocated["runtime"]["enforce_eager"])
        self.assertFalse(collocated["runtime"].get("pd", False))
        shared = next(point for point in points if point["variant"] == "pd-shared")
        self.assertTrue(shared["runtime"]["pd"])
        self.assertEqual(shared["runtime"]["prefill_gpu"], 0)
        self.assertEqual(shared["runtime"]["decode_gpu"], 1)
        self.assertEqual(shared["runtime"]["kv_slot_count"], 2)
        self.assertTrue(shared["runtime"]["prefill_enforce_eager"])
        self.assertFalse(shared["runtime"]["decode_enforce_eager"])

    def test_chunked_validation_has_oversize_prompt_and_no_pd_variant(self):
        from benchmarks.suite import expand_suite

        suite = json.loads(
            (ROOT / "benchmarks/suites/pd-strong-baseline-chunked-validation.json").read_text()
        )
        point = expand_suite(suite)[0]
        classes = point["workload"]["classes"]
        long_request = next(item for item in classes if item["name"] == "long")

        self.assertTrue(point["runtime"]["enable_chunked_prefill"])
        self.assertFalse(point["runtime"].get("pd", False))
        self.assertGreater(
            long_request["input_len"], point["runtime"]["max_num_batched_tokens"]
        )


class StrongBaselineReportTests(unittest.TestCase):
    def test_report_uses_engine_ttft_and_validates_shared_transport(self):
        from benchmarks.pd_strong_baseline import build_report

        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory) / "results"
            runs = results_dir / "runs"
            runs.mkdir(parents=True)
            (results_dir / "manifest.json").write_text(json.dumps({
                "complete": True,
                "git_commit": "abc123",
                "model": "Qwen3-8B",
                "model_revision": "revision",
                "execution_order": [],
            }))
            for variant, pd in (("strong-collocated", False), ("pd-shared", True)):
                for run in range(3):
                    result = _result(variant, run, pd=pd)
                    (runs / f"{result['point_id']}.json").write_text(
                        json.dumps(result)
                    )

            report_path = build_report(results_dir)
            report = report_path.read_text()
            self.assertIn("Engine TTFT", report)
            self.assertIn("not a client-receipt timestamp", report)
            self.assertIn("complete deployment comparison", report)
            self.assertIn(
                "## First-token and PD continuation diagnostics",
                report,
            )
            self.assertIn("Prefill queue wait P50/P99", report)
            self.assertIn("decode_continuation_ready_ms", report)
            self.assertNotIn("TTFT Diagnostic Breakdown", report)
            self.assertIn("## Measurement Window and Sample Rules", report)
            self.assertIn("## Decode-first and CUDA Graph Evidence", report)
            self.assertIn("## Report Revision", report)
            self.assertIn("deterministically from the same per-run seed", report)
            self.assertNotIn("Inline Queue", report)
            timeline_path = results_dir / "request_timeline.csv"
            with timeline_path.open() as input_file:
                rows = list(csv.DictReader(input_file))
            self.assertEqual(len(rows), 6)
            self.assertAlmostEqual(float(rows[0]["engine_ttft_ms"]), 11.0)
            self.assertTrue((results_dir / "pd_handoff.csv").exists())
            self.assertTrue((results_dir / "scheduler_steps.csv").exists())
            trace_audit = json.loads(
                (results_dir / "request_trace_audit.json").read_text()
            )
            self.assertEqual(trace_audit["generator"], "iter_request_specs")
            self.assertEqual(trace_audit["runs"][0]["workload_seed"], 0)
            self.assertEqual(
                trace_audit["runs"][0]["shared_request_count"], 1
            )

    def test_report_rejects_inline_or_unreleased_shared_transport(self):
        from benchmarks.pd_strong_baseline import build_report

        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            runs = results_dir / "runs"
            runs.mkdir()
            (results_dir / "manifest.json").write_text(json.dumps({}))
            for run in range(3):
                baseline = _result("strong-collocated", run)
                (runs / f"baseline-{run}.json").write_text(json.dumps(baseline))
                result = _result("pd-shared", run, pd=True)
                if run == 0:
                    result["metrics"]["pd"]["prefill_batches_detail"][0]["transport"] = "inline"
                    result["metrics"]["pd"]["prefill_batches_detail"][0]["transports"] = ["inline"]
                (runs / f"shared-{run}.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "inline"):
                build_report(results_dir)

    def test_report_distinguishes_enabled_from_actual_main_chunks(self):
        from benchmarks.pd_strong_baseline import build_report

        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory) / "results"
            runs = results_dir / "runs"
            runs.mkdir(parents=True)
            (results_dir / "manifest.json").write_text(json.dumps({
                "complete": True,
                "git_commit": "abc123",
                "model": "Qwen3-8B",
                "model_revision": "revision",
                "execution_order": [],
            }))
            for variant, pd in (("strong-collocated", False), ("pd-shared", True)):
                for run in range(3):
                    result = _result(variant, run, pd=pd)
                    (runs / f"{result['point_id']}.json").write_text(
                        json.dumps(result)
                    )

            validation_dir = Path(directory) / "chunked-validation"
            validation_runs = validation_dir / "runs"
            validation_runs.mkdir(parents=True)
            validation = _result("strong-collocated", 0)
            validation["config"]["workload"]["classes"] = [
                {"name": "short", "weight": 0.8, "input_len": 128, "output_len": 64},
                {"name": "long", "weight": 0.2, "input_len": 2048, "output_len": 64},
            ]
            validation["config"]["runtime"]["max_model_len"] = 2304
            validation["telemetry"]["effective_runtime"] = {
                "kind": "collocated",
                "engine_config": dict(validation["config"]["runtime"]),
            }
            validation["metrics"]["kv_cache"] = {
                "total_blocks": 146,
                "peak_reserved_blocks": 64,
            }
            validation["requests"][0]["request_class"] = "short"
            validation["requests"][0]["engine_ttft_ms"] = 11.0
            validation["telemetry"]["scheduler_steps"][0].update({
                "prefill_chunk_lengths": [768],
                "partial_prefill_chunk_count": 1,
            })
            warmup_request = json.loads(json.dumps(validation["requests"][0]))
            warmup_request["engine_ttft_ms"] = 999.0
            warmup_request["timeline"].update({
                "t_submit": 1.0,
                "t_prefill_first_scheduled": 1.001,
                "t_prefill_finish": 1.011,
                "t_first_token": 1.011,
                "t_handoff_finish": 1.011,
                "t_decode_admitted": 1.011,
                "t_finish": 1.050,
            })
            validation["requests"].append(warmup_request)
            (validation_runs / "validation.json").write_text(json.dumps(validation))

            report = build_report(
                results_dir,
                chunked_results_dir=validation_dir,
            ).read_text()

            self.assertIn(
                "A recorded actual partial chunks per run: **[0, 0, 0]**",
                report,
            )
            self.assertIn("partial chunks: **1**", report)
            self.assertIn("Mixed prefill/decode rounds: **1**", report)
            self.assertIn(
                "Short requests: Engine TTFT P50/P99 = 11.00/11.00 ms",
                report,
            )

    def test_report_formats_multiple_reproduction_commands_as_a_code_block(self):
        from benchmarks.pd_strong_baseline import build_report

        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory) / "results"
            runs = results_dir / "runs"
            runs.mkdir(parents=True)
            (results_dir / "manifest.json").write_text(json.dumps({
                "complete": True,
                "git_commit": "abc123",
                "model": "Qwen3-8B",
                "model_revision": "revision",
                "execution_order": [],
            }))
            for variant, pd in (("strong-collocated", False), ("pd-shared", True)):
                for run in range(3):
                    result = _result(variant, run, pd=pd)
                    (runs / f"{result['point_id']}.json").write_text(
                        json.dumps(result)
                    )

            report = build_report(
                results_dir,
                reproduction_command="first-command\nsecond-command",
            ).read_text()

            self.assertIn(
                "## Reproduction\n\n```bash\nfirst-command\nsecond-command\n```",
                report,
            )


if __name__ == "__main__":
    unittest.main()
