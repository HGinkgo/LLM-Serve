import unittest
from threading import Event
from types import SimpleNamespace

from llmserve.pd.protocol import RequestEnvelope
from llmserve.pd.serving import PDServingEngine
from llmserve.sampling_params import SamplingParams


class FakePDCoordinator:

    def __init__(self):
        self.prefill_calls = []
        self.admit_calls = []
        self.decode_calls = 0
        self.decode_called = Event()
        self.second_prefill_started = Event()
        self.allow_second_prefill = Event()
        self.health = {"prefill": {"alive": True}, "decode": {"alive": True}}

    def start(self):
        return None

    def prefill_batch(self, envelopes):
        self.prefill_calls.append(list(envelopes))
        if len(self.prefill_calls) == 2:
            self.second_prefill_started.set()
            if not self.allow_second_prefill.wait(timeout=2):
                raise AssertionError("second prefill did not overlap decode")
        timing = {
            "worker_total_ms": 12.0,
            "model_forward_ms": 8.0,
            "kv_export_copy_ms": 2.0,
            "forward_calls": 1,
        }
        return [
            SimpleNamespace(
                request_id=envelope.request_id,
                prefill_timing_ms=timing,
            )
            for envelope in envelopes
        ]

    def admit_batch(self, handoffs):
        self.admit_calls.append(list(handoffs))
        return [
            {
                "seq_id": 100 + handoff.request_id,
                "finished": False,
                "output_token_ids": None,
            }
            for handoff in handoffs
        ]

    def decode_step(self):
        self.decode_calls += 1
        self.decode_called.set()
        self.allow_second_prefill.set()
        if self.decode_calls == 1 and len(self.admit_calls[0]) == 2:
            return {
                "outputs": [(100, [200]), (101, [201])],
                "num_tokens": 2,
                "last_step_events": {
                    "step_end": 1.0,
                    "scheduled_seq_ids": [100, 101],
                    "waiting_queue_size": 0,
                    "running_queue_size": 2,
                },
            }
        request_id = self.decode_calls - 1
        return {
            "outputs": [(100 + request_id, [200 + request_id])],
            "num_tokens": 1,
            "last_step_events": {
                "step_end": float(self.decode_calls),
                "scheduled_seq_ids": [100 + request_id],
                "waiting_queue_size": 0,
                "running_queue_size": 1,
            },
        }

    def decode_metrics(self):
        return {
            "requests": [
                {
                    "seq_id": 100,
                    "prompt_tokens": 2,
                    "output_tokens": 1,
                    "success": True,
                    "failure_reason": None,
                    "arrival_time": 10.0,
                    "first_token_time": 11.0,
                    "token_times": [11.0],
                    "output_event_times": [11.0],
                    "finish_time": 11.0,
                },
            ],
            "summary": {"kv_cache": {"total_blocks": 12}},
        }

    def reset_decode_metrics(self):
        return None

    def worker_health(self):
        return self.health

    def close(self):
        return None


class TestPDServingEngine(unittest.TestCase):

    @staticmethod
    def envelope(request_id):
        return RequestEnvelope(
            request_id=request_id,
            prompt_token_ids=(request_id + 1, request_id + 2),
            max_tokens=2,
            temperature=1.0,
            ignore_eos=True,
        )

    def test_step_batches_handoff_and_returns_finished_request_ids(self):
        coordinator = FakePDCoordinator()
        engine = PDServingEngine(coordinator, prefill_batch_size=2)

        first = engine.add_request([1, 2], SamplingParams(max_tokens=2))
        second = engine.add_request([3, 4], SamplingParams(max_tokens=2))
        outputs, num_tokens = engine.step()

        self.assertEqual((first, second), (0, 1))
        self.assertEqual(outputs, [(0, [200]), (1, [201])])
        self.assertEqual(num_tokens, 2)
        self.assertEqual(len(coordinator.prefill_calls), 1)
        self.assertEqual(len(coordinator.admit_calls), 1)
        self.assertEqual(coordinator.decode_calls, 1)
        self.assertTrue(engine.is_finished())

    def test_next_prefill_overlaps_decode_step(self):
        coordinator = FakePDCoordinator()
        engine = PDServingEngine(coordinator, prefill_batch_size=1)

        engine.add_request([1, 2], SamplingParams(max_tokens=2))
        engine.add_request([3, 4], SamplingParams(max_tokens=2))

        outputs, _ = engine.step()

        self.assertEqual(outputs, [(0, [200])])
        self.assertTrue(coordinator.second_prefill_started.wait(timeout=2))
        self.assertTrue(coordinator.decode_called.is_set())

    def test_get_metrics_maps_decode_worker_sequences_to_request_ids(self):
        coordinator = FakePDCoordinator()
        engine = PDServingEngine(coordinator, prefill_batch_size=1)
        engine.add_request([1, 2], SamplingParams(max_tokens=1))
        arrival_time = engine._requests[0]["arrival_time"]
        engine.step()

        metrics = engine.get_metrics()

        self.assertEqual(metrics["requests"][0]["seq_id"], 0)
        self.assertEqual(metrics["requests"][0]["arrival_time"], arrival_time)
        self.assertEqual(metrics["summary"]["kv_cache"]["total_blocks"], 12)
        self.assertIn("pd", metrics["summary"])

    def test_get_metrics_includes_prefill_worker_timing_breakdown(self):
        coordinator = FakePDCoordinator()
        engine = PDServingEngine(coordinator, prefill_batch_size=1)
        engine.add_request([1, 2], SamplingParams(max_tokens=1))
        engine.step()

        timing = engine.get_metrics()["summary"]["pd"]["prefill_batches_detail"][0]

        self.assertEqual(timing["prefill_worker_ms"], 12.0)
        self.assertEqual(timing["prefill_model_forward_ms"], 8.0)
        self.assertEqual(timing["prefill_kv_export_copy_ms"], 2.0)
        self.assertEqual(timing["prefill_forward_calls"], 1)
        self.assertEqual(
            engine.get_metrics()["summary"]["pd"]["prefill_timing"]["worker_ms"],
            12.0,
        )


if __name__ == "__main__":
    unittest.main()
