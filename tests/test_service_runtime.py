import time
import unittest

from llmserve.sampling_params import SamplingParams


class FakeTokenizer:
    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join(chr(token_id) for token_id in token_ids)


class FakeEngine:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.next_request_id = 0
        self.active = {}
        self.abort_calls = []
        self.last_step_events = {}
        self.exited = False

    def add_request(self, prompt, sampling_params):
        request_id = self.next_request_id
        self.next_request_id += 1
        self.active[request_id] = {
            "tokens": [
                ord("a") + index % 3
                for index in range(sampling_params.max_tokens)
            ],
            "index": 0,
            "max_tokens": sampling_params.max_tokens,
        }
        return request_id

    def abort_request(self, request_id):
        self.abort_calls.append(request_id)
        return self.active.pop(request_id, None) is not None

    def is_finished(self):
        return not self.active

    def step(self):
        emitted = {}
        outputs = []
        finished = []
        for request_id, state in list(self.active.items()):
            token_id = state["tokens"][state["index"]]
            state["index"] += 1
            emitted[request_id] = [token_id]
            if state["index"] == len(state["tokens"]):
                outputs.append((request_id, state["tokens"]))
                finished.append(request_id)
                del self.active[request_id]
        self.last_step_events = {
            "emitted_token_ids_by_seq": emitted,
            "finished_seq_ids": finished,
            "waiting_queue_size": 0,
            "running_queue_size": len(self.active),
        }
        time.sleep(0.002)
        return outputs, sum(len(token_ids) for token_ids in emitted.values())

    def get_metrics(self):
        return {
            "summary": {
                "ttft": {"p50": 0.01, "p99": 0.02},
                "tpot": {"p50": 0.003, "p99": 0.004},
                "request_latency": {"p50": 0.02, "p99": 0.03},
            }
        }

    def exit(self):
        self.exited = True


class FakePDEngine(FakeEngine):
    def __init__(self):
        super().__init__()
        del self.tokenizer


class StalledEngine(FakeEngine):
    """Accept requests without emitting output, to exercise service limits."""

    def step(self):
        self.last_step_events = {
            "emitted_token_ids_by_seq": {},
            "finished_seq_ids": [],
            "waiting_queue_size": 0,
            "running_queue_size": len(self.active),
        }
        time.sleep(0.002)
        return [], 0


class FailingStepEngine(FakeEngine):
    """Raise after admission to exercise the driver fatal-error path."""

    def step(self):
        raise RuntimeError("synthetic engine failure")


class ServiceRuntimeTests(unittest.TestCase):
    def setUp(self):
        from llmserve.service.runtime import EngineServiceRuntime

        self.engine = FakeEngine()
        self.runtime = EngineServiceRuntime(lambda: self.engine)
        self.runtime.start()

    def tearDown(self):
        self.runtime.close()

    def test_streams_tokens_before_completion(self):
        request = self.runtime.submit(
            [1, 2], SamplingParams(temperature=0.01, max_tokens=3)
        )

        events = list(request.iter_events(timeout=1.0))

        self.assertEqual([event.kind for event in events], [
            "token", "token", "token", "completed",
        ])
        self.assertEqual([event.text for event in events[:3]], ["a", "b", "c"])
        self.assertEqual(events[-1].token_ids, (ord("a"), ord("b"), ord("c")))
        self.assertTrue(self.runtime.ready)

    def test_poll_event_returns_available_event_without_blocking(self):
        request = self.runtime.submit(
            [1, 2], SamplingParams(temperature=0.01, max_tokens=1)
        )

        deadline = time.monotonic() + 1.0
        event = None
        while event is None and time.monotonic() < deadline:
            event = request.poll_event()
            if event is None:
                time.sleep(0.001)

        self.assertIsNotNone(event)
        self.assertEqual(event.kind, "token")
        deadline = time.monotonic() + 1.0
        terminal = None
        while terminal is None and time.monotonic() < deadline:
            terminal = request.poll_event()
            if terminal is None:
                time.sleep(0.001)
        self.assertIsNotNone(terminal)
        self.assertEqual(terminal.kind, "completed")
        self.assertIsNone(request.poll_event())

    def test_cancel_is_applied_by_the_engine_driver(self):
        request = self.runtime.submit(
            [1, 2], SamplingParams(temperature=0.01, max_tokens=64)
        )
        first_event = next(request.iter_events(timeout=1.0))

        self.assertEqual(first_event.kind, "token")
        self.assertTrue(self.runtime.cancel(request))
        events = list(request.iter_events(timeout=1.0))

        self.assertEqual(events[-1].kind, "cancelled")
        self.assertNotIn("completed", [event.kind for event in events])
        self.assertEqual(self.engine.abort_calls, [request.engine_request_id])

    def test_close_releases_the_underlying_engine(self):
        self.runtime.close()

        self.assertTrue(self.engine.exited)
        self.assertFalse(self.runtime.ready)

    def test_uses_explicit_tokenizer_when_engine_does_not_expose_one(self):
        from llmserve.service.runtime import EngineServiceRuntime

        engine = FakePDEngine()
        runtime = EngineServiceRuntime(lambda: engine, tokenizer=FakeTokenizer())
        runtime.start()
        self.addCleanup(runtime.close)

        request = runtime.submit(
            [1, 2], SamplingParams(temperature=0.01, max_tokens=1)
        )
        events = list(request.iter_events(timeout=1.0))

        self.assertEqual([event.kind for event in events], ["token", "completed"])
        self.assertEqual(events[0].text, "a")

    def test_rejects_submission_when_inflight_limit_is_reached(self):
        from llmserve.service.runtime import EngineServiceRuntime, ServiceRuntimeError

        engine = StalledEngine()
        runtime = EngineServiceRuntime(
            lambda: engine,
            max_inflight_requests=1,
        )
        runtime.start()
        self.addCleanup(runtime.close)

        first = runtime.submit(
            [1], SamplingParams(temperature=0.01, max_tokens=8)
        )
        with self.assertRaisesRegex(ServiceRuntimeError, "in-flight limit"):
            runtime.submit([2], SamplingParams(temperature=0.01, max_tokens=8))

        snapshot = runtime.metrics_snapshot()
        self.assertEqual(first.engine_request_id, 0)
        self.assertEqual(snapshot["inflight_requests"], 1)
        self.assertEqual(snapshot["rejected_requests"], 1)
        self.assertEqual(snapshot["inflight_high_watermark"], 1)

    def test_deadline_cancels_request_and_releases_its_admission_slot(self):
        from llmserve.service.runtime import EngineServiceRuntime

        engine = StalledEngine()
        runtime = EngineServiceRuntime(
            lambda: engine,
            request_timeout_seconds=0.01,
            max_inflight_requests=1,
        )
        runtime.start()
        self.addCleanup(runtime.close)

        request = runtime.submit(
            [1], SamplingParams(temperature=0.01, max_tokens=8)
        )
        terminal = list(request.iter_events(timeout=1.0))[-1]
        snapshot = runtime.metrics_snapshot()

        self.assertEqual(terminal.kind, "timed_out")
        self.assertEqual(engine.abort_calls, [request.engine_request_id])
        self.assertEqual(snapshot["timed_out_requests"], 1)
        self.assertEqual(snapshot["inflight_requests"], 0)

    def test_driver_failure_preserves_root_cause_for_status_and_submit(self):
        from llmserve.service.runtime import EngineServiceRuntime, ServiceRuntimeError

        engine = FailingStepEngine()
        runtime = EngineServiceRuntime(
            lambda: engine,
            submit_timeout_seconds=0.05,
        )
        runtime.start()
        self.addCleanup(runtime.close)

        request = runtime.submit(
            [1], SamplingParams(temperature=0.01, max_tokens=1)
        )
        terminal = next(request.iter_events(timeout=1.0))

        self.assertEqual(terminal.kind, "failed")
        self.assertIn("synthetic engine failure", terminal.message)
        self.assertFalse(runtime.ready)
        self.assertIsInstance(runtime.fatal_error, RuntimeError)
        self.assertEqual(str(runtime.fatal_error), "synthetic engine failure")

        snapshot = runtime.metrics_snapshot()

        self.assertEqual(snapshot["fatal_error"], {
            "type": "RuntimeError",
            "message": "synthetic engine failure",
        })
        self.assertEqual(snapshot["inflight_requests"], 0)
        with self.assertRaisesRegex(ServiceRuntimeError, "synthetic engine failure"):
            runtime.submit([2], SamplingParams(temperature=0.01, max_tokens=1))


if __name__ == "__main__":
    unittest.main()
