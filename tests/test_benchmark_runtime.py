import importlib.util
import unittest

from benchmarks.workloads import RequestSpec


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def perf_counter(self):
        return self.now

    def sleep(self, duration):
        self.sleeps.append(duration)
        self.now += duration


class FakeEngine:
    def __init__(self, clock):
        self.clock = clock
        self.active = []
        self.requests = []
        self.last_step_events = {}

    def add_request(self, prompt, sampling_params):
        seq_id = len(self.requests)
        self.active.append(seq_id)
        self.requests.append({
            "seq_id": seq_id,
            "prompt_tokens": len(prompt),
            "output_tokens": 0,
            "success": False,
            "failure_reason": None,
            "arrival_time": self.clock.now,
            "first_token_time": None,
            "token_times": [],
            "output_event_times": [],
            "finish_time": None,
        })
        return seq_id

    def is_finished(self):
        return not self.active

    def step(self):
        scheduled = list(self.active)
        self.clock.now += 0.25
        seq_id = self.active.pop(0)
        request = self.requests[seq_id]
        request["output_tokens"] = 1
        request["success"] = True
        request["first_token_time"] = self.clock.now
        request["token_times"] = [self.clock.now]
        request["output_event_times"] = [self.clock.now]
        request["finish_time"] = self.clock.now
        self.last_step_events = {
            "step_end": self.clock.now,
            "scheduled_seq_ids": scheduled,
            "waiting_queue_size": max(len(self.active) - 1, 0),
            "running_queue_size": len(self.active),
        }
        return [(seq_id, [1])], 1

    def get_metrics(self):
        return {"requests": self.requests, "summary": {"speculative": {"steps": 0}}}


class BenchmarkRuntimeTests(unittest.TestCase):
    def test_scheduler_telemetry_keeps_replica_local_events(self):
        from benchmarks.runtime import _compact_scheduler_step

        compact = _compact_scheduler_step({
            "replica_events": {
                "replica-0": {"prefill_request_count": 1},
                "replica-1": {"decode_request_count": 1},
            },
            "replica_queue_state": {"replica-0": {"running_queue_size": 1}},
            "scheduled_seq_ids": [0, 1],
        })

        self.assertEqual(compact["scheduled_request_count"], 2)
        self.assertEqual(compact["replica_events"]["replica-0"]["prefill_request_count"], 1)

    def test_submit_uses_benchmark_submission_boundary_when_engine_supports_it(self):
        from benchmarks.runtime import _submit_request

        class BenchmarkBoundaryEngine(FakeEngine):
            def __init__(self, clock):
                super().__init__(clock)
                self.benchmark_submissions = []

            def add_benchmark_request(self, prompt, sampling_params, submitted_at):
                self.benchmark_submissions.append((tuple(prompt), sampling_params, submitted_at))
                return 71

            def add_request(self, prompt, sampling_params):
                raise AssertionError("benchmark path must not use add_request")

        clock = FakeClock()
        engine = BenchmarkBoundaryEngine(clock)

        seq_id = _submit_request(engine, [1, 2], "params", clock.perf_counter)

        self.assertEqual(seq_id, 71)
        self.assertEqual(engine.benchmark_submissions, [((1, 2), "params", 0.0)])
    def test_poisson_runner_submits_on_schedule_and_labels_requests(self):
        self.assertIsNotNone(importlib.util.find_spec("benchmarks.runtime"))
        from benchmarks.runtime import run_poisson

        clock = FakeClock()
        engine = FakeEngine(clock)
        specs = [
            RequestSpec(0, "short", 2, 1, (1, 2)),
            RequestSpec(1, "long", 3, 1, (3, 4, 5)),
            RequestSpec(2, "short", 2, 1, (6, 7)),
        ]

        observation = run_poisson(
            engine,
            specs,
            arrival_times=[0.0, 0.5, 1.0],
            make_sampling_params=lambda spec: spec.output_len,
            clock=clock.perf_counter,
            sleep=clock.sleep,
        )

        self.assertEqual(observation["admitted"], 3)
        self.assertEqual(observation["duration"], 1.25)
        self.assertEqual(
            [request["request_class"] for request in observation["requests"]],
            ["short", "long", "short"],
        )
        self.assertEqual(observation["scheduled_batch_sizes"], [1, 1, 1])
        self.assertEqual(observation["waiting_queue_sizes"], [0, 0, 0])
        self.assertTrue(clock.sleeps)

    def test_poisson_runner_rejects_mismatched_arrivals(self):
        self.assertIsNotNone(importlib.util.find_spec("benchmarks.runtime"))
        from benchmarks.runtime import run_poisson

        with self.assertRaisesRegex(ValueError, "arrival_times"):
            run_poisson(
                FakeEngine(FakeClock()),
                [RequestSpec(0, "short", 1, 1, (1,))],
                arrival_times=[],
                make_sampling_params=lambda spec: spec.output_len,
            )

    def test_poisson_runner_counts_delay_between_scheduled_and_admitted_time(self):
        from benchmarks.runtime import run_poisson

        clock = FakeClock()
        engine = FakeEngine(clock)
        specs = [
            RequestSpec(0, "short", 1, 1, (1,)),
            RequestSpec(1, "short", 1, 1, (2,)),
        ]

        observation = run_poisson(
            engine,
            specs,
            arrival_times=[0.0, 0.1],
            make_sampling_params=lambda spec: spec.output_len,
            clock=clock.perf_counter,
            sleep=clock.sleep,
        )

        self.assertEqual(observation["requests"][1]["arrival_time"], 0.1)
        self.assertAlmostEqual(
            observation["requests"][1]["first_token_time"]
            - observation["requests"][1]["arrival_time"],
            0.4,
        )
        self.assertEqual(observation["speculative_batch_sizes"], [])

    def test_closed_loop_refills_during_window_then_drains(self):
        import benchmarks.runtime as runtime

        self.assertTrue(hasattr(runtime, "run_closed_loop"))
        run_closed_loop = runtime.run_closed_loop

        clock = FakeClock()
        engine = FakeEngine(clock)
        specs = (
            RequestSpec(index, "short", 2, 1, (index, index + 1))
            for index in range(10)
        )

        observation = run_closed_loop(
            engine,
            specs,
            max_concurrency=2,
            warmup_seconds=0.5,
            measurement_seconds=1.5,
            make_sampling_params=lambda spec: spec.output_len,
            clock=clock.perf_counter,
        )

        self.assertEqual(observation["duration"], 1.5)
        self.assertEqual(observation["admitted"], 9)
        self.assertEqual(observation["window_completed"], 6)
        self.assertEqual(observation["window_output_tokens"], 6)
        self.assertEqual(len(observation["latency_requests"]), 4)
        self.assertEqual(observation["scheduled_batch_sizes"], [2] * 6)
        self.assertEqual(observation["speculative_batch_sizes"], [])
        self.assertTrue(engine.is_finished())

    def test_closed_loop_emits_auditable_request_trace_without_prompt_tokens(self):
        from benchmarks.runtime import run_closed_loop

        clock = FakeClock()
        engine = FakeEngine(clock)
        specs = (
            RequestSpec(index, "short", 2, 1, (index, index + 1))
            for index in range(8)
        )

        observation = run_closed_loop(
            engine,
            specs,
            max_concurrency=2,
            warmup_seconds=0,
            measurement_seconds=1,
            make_sampling_params=lambda spec: spec.output_len,
            clock=clock.perf_counter,
        )

        trace = observation["request_trace"]
        self.assertEqual(trace["entry_count"], observation["admitted"])
        self.assertTrue(trace["sha256"])
        self.assertEqual(
            [entry["request_id"] for entry in trace["entries"]],
            list(range(observation["admitted"])),
        )
        self.assertNotIn("prompt_token_ids", trace["entries"][0])


class BenchmarkSchemaTests(unittest.TestCase):
    def test_compact_request_removes_absolute_timestamps(self):
        self.assertIsNotNone(importlib.util.find_spec("benchmarks.schema"))
        from benchmarks.schema import compact_request_record

        request = {
            "seq_id": 7,
            "replica_id": "replica-1",
            "request_class": "short",
            "prompt_tokens": 10,
            "output_tokens": 3,
            "success": True,
            "failure_reason": None,
            "arrival_time": 5.0,
            "first_token_time": 6.0,
            "token_times": [6.0, 7.0, 8.0],
            "output_event_times": [6.0, 7.0, 8.0],
            "speculative_step_latency": [0.1, 0.2],
            "finish_time": 8.0,
            "speculative_trace": [{"draft_token_ids": [1, 2, 3]}],
            "speculative_steps": 2,
            "speculative_draft_tokens": 6,
            "speculative_accepted_tokens": 4,
            "speculative_emitted_tokens": 6,
            "speculative_accept_all_count": 1,
            "speculative_gamma_counts": {3: 2},
            "speculative_timing": {"draft": 0.3},
        }

        compact = compact_request_record(request)

        self.assertEqual(compact["seq_id"], 7)
        self.assertEqual(compact["replica_id"], "replica-1")
        self.assertEqual(compact["request_class"], "short")
        self.assertFalse(compact["cancelled"])
        self.assertEqual(compact["status"], "completed")
        self.assertEqual(compact["ttft_ms"], 1000.0)
        self.assertEqual(compact["tpot_ms"], 1000.0)
        self.assertEqual(compact["e2e_ms"], 3000.0)
        self.assertEqual(compact["burst_itl_ms"]["p50"], 1000.0)
        self.assertEqual(compact["output_event_latency_ms"]["p50"], 1000.0)
        self.assertAlmostEqual(compact["speculative_step_latency_ms"]["p50"], 150.0)
        self.assertNotIn("arrival_time", compact)
        self.assertNotIn("token_times", compact)
        self.assertNotIn("output_event_times", compact)
        self.assertNotIn("speculative_trace", compact)
        self.assertEqual(compact["speculative"]["steps"], 2)
        self.assertEqual(compact["speculative"]["gamma_counts"], {3: 2})
        self.assertAlmostEqual(
            compact["speculative"]["timing_ms"]["draft"], 300.0
        )


if __name__ == "__main__":
    unittest.main()
