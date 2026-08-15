import unittest
from threading import Event, Thread
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
        self.close_calls = 0
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
        self.close_calls += 1


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
        engine = PDServingEngine(
            coordinator,
            prefill_batch_size=1,
            enable_latency_telemetry=True,
        )

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

    def test_metrics_classify_decode_idle_while_waiting_for_prefill(self):
        class BlockingPrefillCoordinator(FakePDCoordinator):
            def __init__(self):
                super().__init__()
                self.prefill_started = Event()
                self.allow_prefill = Event()

            def prefill_batch(self, envelopes, release_transfer_ids=()):
                self.prefill_started.set()
                if not self.allow_prefill.wait(timeout=2):
                    raise AssertionError("prefill was not released")
                return super().prefill_batch(
                    envelopes,
                    release_transfer_ids=release_transfer_ids,
                )

        coordinator = BlockingPrefillCoordinator()
        engine = PDServingEngine(
            coordinator,
            prefill_batch_size=1,
            enable_latency_telemetry=True,
        )
        engine.add_request([1, 2], SamplingParams(max_tokens=1))

        worker = Thread(target=engine.step)
        worker.start()
        self.assertTrue(coordinator.prefill_started.wait(timeout=2))
        coordinator.allow_prefill.set()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())

        decode_idle = engine.get_metrics()["summary"]["pd"]["decode_idle"]
        self.assertGreater(
            decode_idle["counts"]["waiting_prefill_output"],
            0,
        )
        self.assertGreaterEqual(
            decode_idle["duration_ms"]["waiting_prefill_output"],
            0.0,
        )

    def test_metrics_preserve_pd_transport_timeline_boundaries(self):
        class TimelineCoordinator(FakePDCoordinator):
            def prefill_batch(self, envelopes, release_transfer_ids=()):
                return [
                    SimpleNamespace(
                        request_id=envelope.request_id,
                        prefill_timing_ms={},
                        telemetry={
                            "t_slot_acquired": 1.001,
                            "t_kv_export_started": 1.002,
                            "t_kv_export_finished": 1.003,
                            "t_slot_ready": 1.004,
                            "t_prefill_first_scheduled": 1.005,
                            "t_prefill_finish": 1.006,
                            "t_first_token": 1.006,
                            "writers": {
                                "t_slot_ready": "prefill_worker.slot_pool",
                            },
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
                        "telemetry": {
                            "t_decode_descriptor_received": 1.007,
                            "t_slot_consuming": 1.0075,
                            "t_decode_admission_started": 1.008,
                            "t_decode_admitted": 1.009,
                            "t_handoff_finish": 1.010,
                            "t_decode_admission_returned": 1.011,
                            "writers": {
                                "t_decode_admitted": "decode_worker.scheduler",
                            },
                        },
                    }
                    for handoff in handoffs
                ]

        engine = PDServingEngine(
            TimelineCoordinator(),
            prefill_batch_size=1,
            enable_latency_telemetry=True,
        )
        engine.add_request([1, 2], SamplingParams(max_tokens=1))
        engine.step()

        timeline = engine.get_metrics()["requests"][0]["timeline"]
        for key in (
            "t_slot_acquired",
            "t_kv_export_started",
            "t_kv_export_finished",
            "t_slot_ready",
            "t_decode_descriptor_received",
            "t_slot_consuming",
            "t_decode_admission_started",
            "t_decode_admitted",
            "t_handoff_finish",
            "t_decode_admission_returned",
        ):
            self.assertIn(key, timeline)
        self.assertEqual(
            timeline["writers"]["t_slot_ready"],
            "prefill_worker.slot_pool",
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

    def test_prefill_failure_marks_every_request_failed_and_closes_engine(self):
        class FailingPrefillCoordinator(FakePDCoordinator):
            def prefill_batch(self, envelopes, release_transfer_ids=()):
                raise RuntimeError("prefill exploded")

        coordinator = FailingPrefillCoordinator()
        engine = PDServingEngine(coordinator, prefill_batch_size=2)
        engine.add_request([1, 2], SamplingParams(max_tokens=2))
        engine.add_request([3, 4], SamplingParams(max_tokens=2))

        with self.assertRaisesRegex(RuntimeError, "prefill exploded"):
            engine.step()

        self.assertTrue(engine.is_finished())
        self.assertEqual(coordinator.close_calls, 1)
        engine.exit()
        self.assertEqual(coordinator.close_calls, 1)
        metrics = engine.get_metrics()
        self.assertEqual(metrics["summary"]["num_failed"], 2)
        self.assertEqual(metrics["summary"]["num_cancelled"], 0)
        self.assertTrue(
            all(request["status"] == "failed" for request in metrics["requests"])
        )
        self.assertTrue(
            all(
                "prefill exploded" in request["failure_reason"]
                for request in metrics["requests"]
            )
        )
        with self.assertRaisesRegex(RuntimeError, "failed"):
            engine.add_request([5, 6], SamplingParams(max_tokens=2))

    def test_decode_failure_clears_active_ownership_and_records_failure(self):
        class FailingDecodeCoordinator(FakePDCoordinator):
            def decode_step(self):
                raise RuntimeError("decode exploded")

        coordinator = FailingDecodeCoordinator()
        engine = PDServingEngine(coordinator, prefill_batch_size=1)
        engine.add_request([1, 2], SamplingParams(max_tokens=2))

        with self.assertRaisesRegex(RuntimeError, "decode exploded"):
            engine.step()

        self.assertEqual(engine._active_by_decode_seq, {})
        self.assertTrue(engine.is_finished())
        request = engine.get_metrics()["requests"][0]
        self.assertEqual(request["status"], "failed")
        self.assertIn("decode exploded", request["failure_reason"])

    def test_admission_failure_releases_unconsumed_shared_transfer(self):
        class FailingAdmissionCoordinator(FakePDCoordinator):
            def prefill_batch(self, envelopes, release_transfer_ids=()):
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

            def admit_batch(self, handoffs):
                raise RuntimeError("admission exploded")

        coordinator = FailingAdmissionCoordinator()
        engine = PDServingEngine(coordinator, prefill_batch_size=1)
        engine.add_request([1, 2], SamplingParams(max_tokens=2))

        with self.assertRaisesRegex(RuntimeError, "admission exploded"):
            engine.step()

        self.assertEqual(coordinator.explicit_release_calls, [["transfer-0"]])
        self.assertTrue(engine.is_finished())

    def test_transfer_release_failure_does_not_mask_admission_failure(self):
        class FailingCleanupCoordinator(FakePDCoordinator):
            def prefill_batch(self, envelopes, release_transfer_ids=()):
                return [
                    SimpleNamespace(
                        request_id=envelopes[0].request_id,
                        descriptor=SimpleNamespace(
                            transport="shared_slot",
                            transfer_id="transfer-0",
                            slot_id=0,
                            slot_generation=1,
                        ),
                        prefill_timing_ms={},
                    )
                ]

            def admit_batch(self, handoffs):
                raise RuntimeError("admission exploded")

            def release_prefill_transfers(self, transfer_ids):
                raise RuntimeError("release exploded")

        coordinator = FailingCleanupCoordinator()
        engine = PDServingEngine(coordinator, prefill_batch_size=1)
        engine.add_request([1, 2], SamplingParams(max_tokens=2))

        with self.assertRaisesRegex(RuntimeError, "admission exploded"):
            engine.step()

        self.assertTrue(engine.is_finished())

    def test_close_failure_does_not_mask_worker_failure(self):
        class FailingPrefillAndCloseCoordinator(FakePDCoordinator):
            def prefill_batch(self, envelopes, release_transfer_ids=()):
                raise RuntimeError("prefill exploded")

            def close(self):
                self.close_calls += 1
                raise RuntimeError("close exploded")

        coordinator = FailingPrefillAndCloseCoordinator()
        engine = PDServingEngine(coordinator, prefill_batch_size=1)
        engine.add_request([1, 2], SamplingParams(max_tokens=2))

        with self.assertRaisesRegex(RuntimeError, "prefill exploded"):
            engine.step()

        self.assertTrue(engine.is_finished())
        self.assertEqual(coordinator.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
