import unittest
from threading import Event
from types import SimpleNamespace

from llmserve.pd.protocol import RequestEnvelope
from llmserve.pd.serving import PDServingEngine
from llmserve.sampling_params import SamplingParams


class FakePDCoordinator:

    def __init__(self):
        self.prefill_calls = []
        self.prefill_release_calls = []
        self.explicit_release_calls = []
        self.admit_calls = []
        self.decode_calls = 0
        self.abort_calls = []
        self.decode_called = Event()
        self.second_prefill_started = Event()
        self.allow_second_prefill = Event()
        self.health = {"prefill": {"alive": True}, "decode": {"alive": True}}

    def start(self):
        return None

    def prefill_batch(self, envelopes, release_transfer_ids=()):
        self.prefill_calls.append(list(envelopes))
        self.prefill_release_calls.append(list(release_transfer_ids))
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

    def release_prefill_transfers(self, transfer_ids):
        self.explicit_release_calls.append(list(transfer_ids))
        return {"free_slots": 2}

    def last_rpc_timing(self, role):
        if role == "prefill":
            return {
                "roundtrip_ms": 12.0,
                "parent_queue_put_ms": 0.1,
                "command_queue_ms": 0.2,
                "worker_service_ms": 10.0,
                "response_queue_ms": 1.7,
            }
        return {
            "roundtrip_ms": 3.0,
            "parent_queue_put_ms": 0.1,
            "command_queue_ms": 0.2,
            "worker_service_ms": 2.0,
            "response_queue_ms": 0.7,
        }

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

    def abort_decode_request(self, seq_id):
        self.abort_calls.append(seq_id)
        return True

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

    def test_metrics_split_prefill_and_admit_rpc_queue_latency(self):
        coordinator = FakePDCoordinator()
        engine = PDServingEngine(coordinator, prefill_batch_size=1)
        engine.add_request([1, 2], SamplingParams(max_tokens=1))

        engine.step()
        timing = engine.get_metrics()["summary"]["pd"]["prefill_batches_detail"][0]

        self.assertEqual(timing["prefill_command_queue_ms"], 0.2)
        self.assertEqual(timing["prefill_response_queue_ms"], 1.7)
        self.assertEqual(timing["decode_command_queue_ms"], 0.2)
        self.assertEqual(timing["decode_worker_admit_ms"], 2.0)
        self.assertEqual(timing["decode_response_queue_ms"], 0.7)

    def test_shared_transfer_ack_is_piggybacked_then_tail_is_flushed(self):
        class SharedCoordinator(FakePDCoordinator):
            def prefill_batch(self, envelopes, release_transfer_ids=()):
                self.prefill_calls.append(list(envelopes))
                self.prefill_release_calls.append(list(release_transfer_ids))
                return [
                    SimpleNamespace(
                        request_id=envelope.request_id,
                        descriptor=SimpleNamespace(
                            transport="shared_slot",
                            transfer_id=f"transfer-{envelope.request_id}",
                            slot_id=envelope.request_id % 2,
                            slot_generation=1,
                        ),
                        prefill_timing_ms={
                            "worker_total_ms": 12.0,
                            "model_forward_ms": 8.0,
                            "kv_export_copy_ms": 2.0,
                            "forward_calls": 1,
                        },
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
                        "transfer_id": handoff.descriptor.transfer_id,
                    }
                    for handoff in handoffs
                ]

        coordinator = SharedCoordinator()
        engine = PDServingEngine(coordinator, prefill_batch_size=1)
        for token in (1, 2, 3):
            engine.add_request([token], SamplingParams(max_tokens=2))

        while not engine.is_finished():
            engine.step()

        self.assertEqual(
            coordinator.prefill_release_calls,
            [[], [], ["transfer-0"]],
        )
        self.assertEqual(
            coordinator.explicit_release_calls,
            [["transfer-1", "transfer-2"]],
        )

    def test_abort_pending_request_is_idempotent_and_reported(self):
        coordinator = FakePDCoordinator()
        engine = PDServingEngine(coordinator, prefill_batch_size=1)
        request_id = engine.add_request([1, 2], SamplingParams(max_tokens=2))

        self.assertTrue(engine.abort_request(request_id))
        self.assertFalse(engine.abort_request(request_id))
        self.assertFalse(engine.abort_request(request_id + 1000))
        self.assertTrue(engine.is_finished())

        metrics = engine.get_metrics()
        request = metrics["requests"][0]
        self.assertEqual(request["status"], "cancelled")
        self.assertTrue(request["cancelled"])
        self.assertEqual(metrics["summary"]["num_cancelled"], 1)
        self.assertEqual(metrics["summary"]["num_failed"], 0)

    def test_abort_active_decode_request_propagates_to_decode_worker(self):
        class ActiveCoordinator(FakePDCoordinator):
            def decode_step(self):
                self.decode_calls += 1
                return {
                    "outputs": [],
                    "num_tokens": 1,
                    "last_step_events": {
                        "scheduled_seq_ids": [100],
                        "waiting_queue_size": 0,
                        "running_queue_size": 1,
                    },
                }

        coordinator = ActiveCoordinator()
        engine = PDServingEngine(coordinator, prefill_batch_size=1)
        request_id = engine.add_request([1, 2], SamplingParams(max_tokens=2))
        engine.step()

        self.assertTrue(engine.abort_request(request_id))
        self.assertEqual(coordinator.abort_calls, [100])
        self.assertEqual(engine._active_by_decode_seq, {})
        self.assertTrue(engine.is_finished())

    def test_abort_inflight_prefill_discards_handoff_and_releases_slot(self):
        class BlockingSharedCoordinator(FakePDCoordinator):
            def __init__(self):
                super().__init__()
                self.prefill_started = Event()
                self.allow_prefill = Event()

            def prefill_batch(self, envelopes, release_transfer_ids=()):
                self.prefill_calls.append(list(envelopes))
                self.prefill_release_calls.append(list(release_transfer_ids))
                self.prefill_started.set()
                if not self.allow_prefill.wait(timeout=2):
                    raise AssertionError("prefill was not released")
                return [
                    SimpleNamespace(
                        request_id=envelope.request_id,
                        descriptor=SimpleNamespace(
                            transport="shared_slot",
                            transfer_id=f"transfer-{envelope.request_id}",
                            slot_id=0,
                            slot_generation=1,
                        ),
                        prefill_timing_ms={},
                    )
                    for envelope in envelopes
                ]

        coordinator = BlockingSharedCoordinator()
        engine = PDServingEngine(coordinator, prefill_batch_size=1)
        request_id = engine.add_request([1, 2], SamplingParams(max_tokens=2))
        engine._start_prefill()
        self.assertTrue(coordinator.prefill_started.wait(timeout=2))

        self.assertTrue(engine.abort_request(request_id))
        coordinator.allow_prefill.set()
        engine._prefill_future.result(timeout=2)
        engine._collect_prefill()

        self.assertEqual(coordinator.admit_calls, [])
        self.assertEqual(coordinator.explicit_release_calls, [["transfer-0"]])
        self.assertTrue(engine.is_finished())


if __name__ == "__main__":
    unittest.main()
