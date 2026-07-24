import importlib.util
import unittest
from types import SimpleNamespace

import torch

from tests.test_benchmark_runtime import FakeClock, FakeEngine


def make_point(**overrides):
    point = {
        "point_id": "chunked-poisson-baseline-rate-2p0-r1",
        "suite": "serving-v1",
        "experiment": "chunked-poisson",
        "arrival": "poisson",
        "variant": "baseline",
        "run": 0,
        "workload_seed": 0,
        "arrival_seed": 0,
        "request_rate": 2.0,
        "num_requests": 3,
        "workload": {
            "classes": [
                {"name": "short", "weight": 2, "input_len": 2, "output_len": 1},
                {"name": "long", "weight": 1, "input_len": 3, "output_len": 1},
            ]
        },
        "runtime": {
            "enforce_eager": True,
            "enable_chunked_prefill": False,
            "enable_speculative": False,
            "max_model_len": 64,
            "max_num_batched_tokens": 16,
        },
        "slo_ms": {"ttft": 1000, "tpot": 1000, "e2e": 1000},
    }
    point.update(overrides)
    return point


class BenchmarkServeTests(unittest.TestCase):
    def test_run_point_builds_sanitized_poisson_result(self):
        self.assertIsNotNone(importlib.util.find_spec("benchmarks.serve"))
        from benchmarks.serve import run_point

        clock = FakeClock()
        factory_calls = []

        def engine_factory(model, **kwargs):
            factory_calls.append((model, kwargs))
            return FakeEngine(clock)

        result = run_point(
            make_point(),
            model="/private/models/Qwen3-8B",
            speculative_model="/private/models/eagle3",
            distributed_init_method="tcp://localhost:2444",
            engine_factory=engine_factory,
            make_sampling_params=lambda spec: spec.output_len,
            clock=clock.perf_counter,
            sleep=clock.sleep,
            metadata={"git_commit": "abc123", "git_dirty": False},
            model_revision="target-rev",
            speculative_model_revision="draft-rev",
        )

        self.assertEqual(factory_calls[0][0], "/private/models/Qwen3-8B")
        self.assertEqual(
            factory_calls[0][1]["distributed_init_method"],
            "tcp://localhost:2444",
        )
        self.assertIsNone(factory_calls[0][1]["speculative_model"])
        self.assertFalse(factory_calls[0][1]["enable_chunked_prefill"])
        self.assertEqual(result["schema_version"], 2)
        self.assertTrue(result["complete"])
        self.assertEqual(result["git_commit"], "abc123")
        self.assertEqual(result["config"]["model"], "Qwen3-8B")
        self.assertIsNone(result["config"]["speculative_model"])
        self.assertEqual(result["config"]["model_revision"], "target-rev")
        self.assertEqual(result["metrics"]["offered_request_rate"], 2.0)
        self.assertEqual(result["metrics"]["completed"], 3)
        self.assertIn("scheduled_batch_size", result["metrics"])
        self.assertIn("waiting_queue_size", result["metrics"])
        self.assertIn("speculative_batch_size", result["metrics"])
        self.assertEqual(len(result["requests"]), 3)
        self.assertNotIn("token_times", result["requests"][0])

    def test_run_point_requires_draft_path_for_speculative_variant(self):
        self.assertIsNotNone(importlib.util.find_spec("benchmarks.serve"))
        from benchmarks.serve import run_point

        point = make_point()
        point["runtime"]["enable_speculative"] = True

        with self.assertRaisesRegex(ValueError, "speculative_model"):
            run_point(
                point,
                model="/models/Qwen3-8B",
                speculative_model=None,
                engine_factory=lambda *args, **kwargs: None,
            )

    def test_run_point_installs_deterministic_argmax_sampler(self):
        from benchmarks.serve import run_point

        clock = FakeClock()
        engine = FakeEngine(clock)
        engine.model_runner = SimpleNamespace(sampler=None)
        point = make_point()
        point["runtime"]["argmax_sampler"] = True

        run_point(
            point,
            model="/models/Qwen3-8B",
            engine_factory=lambda *args, **kwargs: engine,
            make_sampling_params=lambda spec: spec.output_len,
            clock=clock.perf_counter,
            sleep=clock.sleep,
        )

        logits = torch.tensor([[0.0, 2.0, 1.0]])
        self.assertEqual(engine.model_runner.sampler(logits, None).item(), 1)

    def test_run_point_passes_awq_backend_to_engine(self):
        from benchmarks.serve import run_point

        clock = FakeClock()
        factory_calls = []
        point = make_point()
        point["runtime"]["awq_backend"] = "reference"

        def engine_factory(model, **kwargs):
            factory_calls.append(kwargs)
            return FakeEngine(clock)

        run_point(
            point,
            model="/models/Qwen3-8B-AWQ",
            engine_factory=engine_factory,
            make_sampling_params=lambda spec: spec.output_len,
            clock=clock.perf_counter,
            sleep=clock.sleep,
        )

        self.assertEqual(factory_calls[0]["awq_backend"], "reference")

    def test_run_point_passes_capacity_admission_and_reports_kv_metrics(self):
        from benchmarks.serve import run_point

        class CapacityFakeEngine(FakeEngine):
            def get_metrics(self):
                metrics = super().get_metrics()
                metrics["summary"]["kv_cache"] = {
                    "enabled": True,
                    "total_blocks": 321,
                    "preemptions": 0,
                }
                return metrics

        clock = FakeClock()
        factory_calls = []
        point = make_point()
        point["runtime"].update({
            "enable_kv_capacity_admission": True,
            "max_num_seqs": 128,
            "gpu_memory_utilization": 0.92,
        })

        def engine_factory(model, **kwargs):
            factory_calls.append(kwargs)
            return CapacityFakeEngine(clock)

        result = run_point(
            point,
            model="/models/Qwen3-8B-AWQ",
            engine_factory=engine_factory,
            make_sampling_params=lambda spec: spec.output_len,
            clock=clock.perf_counter,
            sleep=clock.sleep,
        )

        self.assertTrue(factory_calls[0]["enable_kv_capacity_admission"])
        self.assertEqual(factory_calls[0]["max_num_seqs"], 128)
        self.assertEqual(factory_calls[0]["gpu_memory_utilization"], 0.92)
        self.assertEqual(result["metrics"]["kv_cache"]["total_blocks"], 321)
        self.assertEqual(result["metrics"]["kv_cache"]["preemptions"], 0)

    def test_closed_loop_result_reports_latency_sample_request_count(self):
        from benchmarks.serve import run_point

        clock = FakeClock()
        point = make_point(
            point_id="closed-loop-baseline-concurrency-2-r1",
            arrival="closed-loop",
            max_concurrency=2,
            warmup_seconds=0.5,
            measurement_seconds=1.5,
        )

        result = run_point(
            point,
            model="/models/Qwen3-8B",
            engine_factory=lambda *args, **kwargs: FakeEngine(clock),
            make_sampling_params=lambda spec: spec.output_len,
            clock=clock.perf_counter,
            sleep=clock.sleep,
        )

        self.assertEqual(result["metrics"]["completed"], 6)
        self.assertEqual(result["metrics"]["latency_sample_requests"], 4)

    def test_poisson_warmup_is_reset_before_measurement(self):
        from benchmarks.serve import run_point

        class ResettableFakeEngine(FakeEngine):
            def __init__(self, clock):
                super().__init__(clock)
                self.reset_calls = 0

            def reset_metrics(self):
                self.reset_calls += 1
                self.requests.clear()
                self.last_step_events = {}

        clock = FakeClock()
        engine = ResettableFakeEngine(clock)
        point = make_point()
        point["runtime"]["warmup"] = True

        result = run_point(
            point,
            model="/models/Qwen3-8B",
            engine_factory=lambda *args, **kwargs: engine,
            make_sampling_params=lambda spec: spec.output_len,
            clock=clock.perf_counter,
            sleep=clock.sleep,
        )

        self.assertEqual(engine.reset_calls, 1)
        self.assertEqual(result["metrics"]["completed"], 3)
        self.assertEqual(len(result["requests"]), 3)

if __name__ == "__main__":
    unittest.main()
