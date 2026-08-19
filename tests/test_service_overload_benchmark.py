import subprocess
import sys
from threading import Event, Thread
import unittest
from pathlib import Path

from benchmarks.workloads import RequestSpec
from llmserve.service.runtime import GenerationEvent, ServiceOverloadedError


class _ImmediateRequest:
    def __init__(self, request_id: int):
        self._events = [
            GenerationEvent("token", request_id, token_ids=(11,)),
            GenerationEvent("completed", request_id, token_ids=(11,)),
        ]

    def poll_event(self):
        return self._events.pop(0) if self._events else None


class _CapacityOneRuntime:
    def __init__(self):
        self._accepted = 0
        self._rejected = 0
        self._active = 0

    def submit(self, prompt, sampling_params):
        del prompt, sampling_params
        if self._accepted >= 1:
            self._rejected += 1
            raise ServiceOverloadedError("service in-flight limit has been reached")
        request = _ImmediateRequest(self._accepted)
        self._accepted += 1
        self._active = 1
        return request

    def metrics_snapshot(self):
        return {
            "accepted_requests": self._accepted,
            "rejected_requests": self._rejected,
            "inflight_requests": self._active,
            "admission_reserved": self._accepted,
            "queue_waiting": 3,
            "queue_running": 1,
        }

    def cancel(self, request):
        del request
        return True


class _SteppingClock:
    def __init__(self):
        self._value = -1.0

    def __call__(self):
        self._value += 1.0
        return self._value


class _TimedRequest:
    def __init__(self, request_id: int, terminal_kind: str = "completed"):
        self._events = [
            GenerationEvent("token", request_id, token_ids=(11,)),
            GenerationEvent("token", request_id, token_ids=(12, 13)),
            GenerationEvent(terminal_kind, request_id, token_ids=(11, 12, 13)),
        ]

    def poll_event(self):
        return self._events.pop(0) if self._events else None


class _TimedRuntime:
    def __init__(self, terminal_kind: str = "completed"):
        self._terminal_kind = terminal_kind

    def submit(self, prompt, sampling_params):
        del prompt, sampling_params
        return _TimedRequest(0, self._terminal_kind)

    def metrics_snapshot(self):
        return {
            "inflight_requests": 1,
            "admission_reserved": 1,
            "queue_waiting": 7,
            "queue_running": 1,
        }

    def cancel(self, request):
        del request
        return True


class _PromptCapturingRuntime(_TimedRuntime):
    def __init__(self):
        super().__init__()
        self.prompts = []

    def submit(self, prompt, sampling_params):
        self.prompts.append(prompt)
        return super().submit(prompt, sampling_params)


class _BlockingFirstSubmitRuntime(_TimedRuntime):
    def __init__(self):
        super().__init__()
        self.first_submit_started = Event()
        self.second_submit_started = Event()
        self.release_first_submit = Event()
        self.second_started_before_release = False
        self._submit_count = 0

    def submit(self, prompt, sampling_params):
        del prompt, sampling_params
        submit_index = self._submit_count
        self._submit_count += 1
        if submit_index == 0:
            self.first_submit_started.set()
            self.release_first_submit.wait(timeout=0.2)
        else:
            self.second_started_before_release = not self.release_first_submit.is_set()
            self.second_submit_started.set()
        return _ImmediateRequest(submit_index)


class ServiceOverloadBenchmarkTests(unittest.TestCase):
    def test_launches_each_point_in_an_isolated_worker_process(self):
        from benchmarks.service_overload import _run_point_worker

        calls = []

        def runner(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0, "worker output", "")

        result = _run_point_worker(
            Path("/tmp/service-overload-point.json"),
            runner=runner,
        )

        self.assertEqual(result.returncode, 0)
        self.assertEqual(calls[0][0], [
            sys.executable,
            "-m",
            "benchmarks.service_overload",
            "--worker-config",
            "/tmp/service-overload-point.json",
        ])
        self.assertFalse(calls[0][1]["check"])

    def test_parses_unbounded_and_bounded_admission_variants(self):
        from benchmarks.service_overload import parse_inflight_limit_variants

        self.assertEqual(
            parse_inflight_limit_variants("unbounded,64"),
            (("unbounded", None), ("limit-64", 64)),
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            parse_inflight_limit_variants("0")

    def test_records_overload_rejections_and_client_visible_latency(self):
        from benchmarks.service_overload import run_service_poisson

        requests = [
            RequestSpec(0, "short", 2, 1, (1, 2)),
            RequestSpec(1, "short", 2, 1, (3, 4)),
        ]
        observation = run_service_poisson(
            _CapacityOneRuntime(),
            request_specs=requests,
            arrival_times=(0.0, 0.0),
            sampling_params_factory=lambda spec: spec.output_len,
            warmup_seconds=0.0,
            measurement_seconds=0.02,
            drain_timeout_seconds=0.01,
            snapshot_interval_seconds=0.001,
        )

        self.assertEqual(observation["admission"]["offered"], 2)
        self.assertEqual(observation["admission"]["accepted"], 1)
        self.assertEqual(observation["admission"]["rejected"], 1)
        self.assertEqual(observation["metrics"]["completed"], 1)
        self.assertEqual(
            observation["metrics"]["latency"]["overall"]["ttft"]["count"],
            1,
        )
        self.assertEqual(observation["queue_depth"]["waiting"]["max"], 3)
        self.assertEqual(observation["requests"][1]["status"], "rejected")

    def test_copies_immutable_trace_tokens_before_submitting_to_engine(self):
        from benchmarks.service_overload import run_service_poisson

        runtime = _PromptCapturingRuntime()
        run_service_poisson(
            runtime,
            request_specs=[RequestSpec(0, "short", 2, 1, (1, 2))],
            arrival_times=(0.0,),
            sampling_params_factory=lambda spec: spec.output_len,
            warmup_seconds=0.0,
            measurement_seconds=0.02,
            drain_timeout_seconds=0.01,
            snapshot_interval_seconds=0.001,
        )

        self.assertEqual(runtime.prompts, [[1, 2]])

    def test_submits_later_arrivals_while_an_earlier_admission_blocks(self):
        from benchmarks.service_overload import run_service_poisson

        runtime = _BlockingFirstSubmitRuntime()

        def release_first_submit():
            runtime.second_submit_started.wait(timeout=0.05)
            runtime.release_first_submit.set()

        releaser = Thread(target=release_first_submit)
        releaser.start()
        try:
            run_service_poisson(
                runtime,
                request_specs=[
                    RequestSpec(0, "short", 2, 1, (1, 2)),
                    RequestSpec(1, "short", 2, 1, (3, 4)),
                ],
                arrival_times=(0.0, 0.01),
                sampling_params_factory=lambda spec: spec.output_len,
                warmup_seconds=0.0,
                measurement_seconds=0.06,
                drain_timeout_seconds=0.01,
                snapshot_interval_seconds=0.001,
            )
        finally:
            runtime.release_first_submit.set()
            releaser.join(timeout=1.0)

        self.assertTrue(runtime.first_submit_started.is_set())
        self.assertTrue(runtime.second_submit_started.is_set())
        self.assertTrue(runtime.second_started_before_release)

    def test_summary_row_exports_only_windowed_streaming_metrics(self):
        from benchmarks.service_overload import _summary_row

        result = {
            "point_id": "limit-64",
            "trace_hash": "trace",
            "config": {
                "request_rate": 24.0,
                "variant": "limit-64",
                "max_inflight_requests": 64,
            },
            "observation": {
                "admission": {
                    "measurement_offered": 10,
                    "measurement_accepted": 8,
                    "measurement_rejected": 2,
                    "unfinished_after_drain": 0,
                },
                "metrics": {
                    "completed": 8,
                    "throughput": {
                        "requests_per_second": 4.0,
                        "input_tokens_per_second": 128.0,
                        "output_tokens_per_second": 256.0,
                    },
                    "latency": {
                        "overall": {
                            "ttft": {"p50": 0.012, "p99": 0.099},
                            "tpot": {"p50": 0.020, "p99": 0.030},
                        }
                    },
                },
                "queue_depth": {
                    "waiting": {"p99": 4.0},
                    "running": {"p99": 64.0},
                    "inflight": {"p99": 64.0},
                },
            },
        }

        row = _summary_row(result)

        self.assertEqual(row["ttft_p50_ms"], 12.0)
        self.assertEqual(row["tpot_p99_ms"], 30.0)
        self.assertNotIn("input_throughput_tps", row)
        self.assertNotIn("e2e_p50_ms", row)
        self.assertNotIn("e2e_p99_ms", row)
        self.assertNotIn("goodput_rps", row)

    def test_parser_does_not_expose_slo_configuration(self):
        from benchmarks.service_overload import build_parser

        self.assertNotIn("--slo-ms", build_parser().format_help())

    def test_uses_token_events_for_tpot_and_marks_queue_summary_as_measurement_only(self):
        from benchmarks.service_overload import run_service_poisson

        observation = run_service_poisson(
            _TimedRuntime(),
            request_specs=[RequestSpec(0, "short", 2, 3, (1, 2))],
            arrival_times=(0.0,),
            sampling_params_factory=lambda spec: spec.output_len,
            warmup_seconds=0.0,
            measurement_seconds=8.0,
            drain_timeout_seconds=0.0,
            snapshot_interval_seconds=0.25,
            clock=_SteppingClock(),
            sleeper=lambda seconds: None,
        )

        latency = observation["metrics"]["latency"]["overall"]
        self.assertEqual(latency["ttft"]["p50"], 3.0)
        self.assertEqual(latency["tpot"]["p50"], 0.5)
        self.assertEqual(set(latency), {"ttft", "tpot"})
        self.assertEqual(
            set(observation["metrics"]["throughput"]),
            {"requests_per_second", "output_tokens_per_second"},
        )
        self.assertNotIn("goodput", observation["metrics"])
        self.assertEqual(observation["queue_depth"]["window"], "measurement")

    def test_reports_measurement_window_timeout_separately_from_latency_cohort(self):
        from benchmarks.service_overload import run_service_poisson

        observation = run_service_poisson(
            _TimedRuntime("timed_out"),
            request_specs=[RequestSpec(0, "short", 2, 3, (1, 2))],
            arrival_times=(0.0,),
            sampling_params_factory=lambda spec: spec.output_len,
            warmup_seconds=0.0,
            measurement_seconds=8.0,
            drain_timeout_seconds=0.0,
            snapshot_interval_seconds=0.25,
            clock=_SteppingClock(),
            sleeper=lambda seconds: None,
        )

        self.assertEqual(
            observation["outcomes"]["measurement_window"]["timed_out"],
            1,
        )
        self.assertEqual(observation["metrics"]["completed"], 0)


if __name__ == "__main__":
    unittest.main()
