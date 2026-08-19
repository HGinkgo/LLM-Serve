import unittest
from threading import Event, Thread
from types import SimpleNamespace

from llmserve.pd.protocol import RequestEnvelope
from llmserve.pd.protocol import RequestState
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

    def collect_completed_transfers(self, wait=False):
        return []

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
                    "emitted_token_ids_by_seq": {100: [200], 101: [201]},
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
                "emitted_token_ids_by_seq": {100 + request_id: [200 + request_id]},
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


class MultiDecodeCoordinator(FakePDCoordinator):

    decode_worker_ids = ("decode-0", "decode-1")

    def __init__(self):
        super().__init__()
        self.prefill_targets = []
        self.admitted_by_worker = {worker_id: [] for worker_id in self.decode_worker_ids}
        self.reset_all_calls = 0

    def prefill_batch(self, envelopes, release_transfer_ids=()):
        self.prefill_calls.append(list(envelopes))
        self.prefill_release_calls.append(list(release_transfer_ids))
        self.prefill_targets.extend(envelope.target_worker for envelope in envelopes)
        return [
            SimpleNamespace(
                request_id=envelope.request_id,
                envelope=envelope,
                descriptor=SimpleNamespace(
                    transport="inline",
                    target_worker=envelope.target_worker,
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

    def admit_batch(self, handoffs, *, decode_worker_id=None):
        self.admit_calls.append(list(handoffs))
        self.admitted_by_worker[decode_worker_id].extend(
            handoff.request_id for handoff in handoffs
        )
        return [
            {"seq_id": 0, "finished": False, "output_token_ids": None}
            for _ in handoffs
        ]

    def decode_step_all(self, decode_worker_ids=None):
        worker_ids = decode_worker_ids or self.decode_worker_ids
        results = {}
        for index, worker_id in enumerate(worker_ids):
            results[worker_id] = {
                "outputs": [(0, [200 + index])],
                "num_tokens": 1,
                "last_step_events": {
                    "scheduled_seq_ids": [0],
                    "emitted_token_ids_by_seq": {0: [200 + index]},
                    "waiting_queue_size": 0,
                    "running_queue_size": 1,
                },
                "completed_transfers": [],
                "step_diagnostics": {},
            }
        return results

    def collect_completed_transfers(self, *, wait=False, decode_worker_id=None):
        return []

    def decode_metrics_all(self):
        return {
            worker_id: {
                "requests": [
                    {
                        "seq_id": 0,
                        "prompt_tokens": 2,
                        "output_tokens": 1,
                        "success": True,
                        "failure_reason": None,
                        "arrival_time": 1.0,
                        "first_token_time": 2.0,
                        "token_times": [2.0],
                        "output_event_times": [2.0],
                        "finish_time": 2.0,
                    }
                ],
                "summary": {"kv_cache": {"total_blocks": 12}},
            }
            for worker_id in self.decode_worker_ids
        }

    def reset_decode_metrics_all(self):
        self.reset_all_calls += 1
        return {worker_id: {"reset": True} for worker_id in self.decode_worker_ids}


class CombinedStepCoordinator(FakePDCoordinator):

    def __init__(self):
        super().__init__()
        self.combined_handoffs = []

    def decode_step_with_handoffs(self, handoffs):
        self.combined_handoffs.append(list(handoffs))
        return {
            "admissions": [
                {
                    "seq_id": 101,
                    "finished": False,
                    "output_token_ids": None,
                }
            ],
            "outputs": [(100, [200])],
            "num_tokens": 1,
            "completed_transfers": [],
            "last_step_events": {
                "step_end": 1.0,
                "scheduled_seq_ids": [100],
                "waiting_queue_size": 0,
                "running_queue_size": 1,
            },
        }


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
        self.assertEqual(
            engine.last_step_events["emitted_token_ids_by_seq"],
            {first: [200], second: [201]},
        )
        self.assertEqual(len(coordinator.prefill_calls), 1)
        self.assertEqual(len(coordinator.admit_calls), 1)
        self.assertEqual(coordinator.decode_calls, 1)
        self.assertTrue(engine.is_finished())

    def test_decode_pool_routes_requests_and_scopes_duplicate_sequence_ids(self):
        coordinator = MultiDecodeCoordinator()
        engine = PDServingEngine(coordinator, prefill_batch_size=2)
        first = engine.add_request([1, 2], SamplingParams(max_tokens=2))
        second = engine.add_request([3, 4], SamplingParams(max_tokens=2))

        outputs, num_tokens = engine.step()

        self.assertEqual(outputs, [(first, [200]), (second, [201])])
        self.assertEqual(num_tokens, 2)
        self.assertEqual(
            engine.last_step_events["emitted_token_ids_by_seq"],
            {first: [200], second: [201]},
        )
        self.assertEqual(coordinator.prefill_targets, ["decode-0", "decode-1"])
        self.assertEqual(coordinator.admitted_by_worker["decode-0"], [first])
        self.assertEqual(coordinator.admitted_by_worker["decode-1"], [second])
        self.assertEqual(engine._requests[first]["decode_worker_id"], "decode-0")
        self.assertEqual(engine._requests[second]["decode_worker_id"], "decode-1")

    def test_decode_pool_cancellation_targets_the_owning_worker(self):
        class ActivePoolCoordinator(MultiDecodeCoordinator):
            def __init__(self):
                super().__init__()
                self.abort_calls = []

            def decode_step_all(self, decode_worker_ids=None):
                return {
                    worker_id: {
                        "outputs": [],
                        "num_tokens": 1,
                        "last_step_events": {"scheduled_seq_ids": [0]},
                        "completed_transfers": [],
                        "step_diagnostics": {},
                    }
                    for worker_id in (decode_worker_ids or self.decode_worker_ids)
                }

            def abort_decode_request(self, seq_id, *, decode_worker_id=None):
                self.abort_calls.append((decode_worker_id, seq_id))
                return True

        coordinator = ActivePoolCoordinator()
        engine = PDServingEngine(coordinator, prefill_batch_size=2)
        first = engine.add_request([1, 2], SamplingParams(max_tokens=2))
        second = engine.add_request([3, 4], SamplingParams(max_tokens=2))
        engine.step()

        self.assertTrue(engine.abort_request(first))
        self.assertEqual(coordinator.abort_calls, [("decode-0", 0)])
        self.assertIn(("decode-1", 0), engine._active_by_decode_seq)
        self.assertNotIn(("decode-0", 0), engine._active_by_decode_seq)
        self.assertFalse(engine._requests[second]["lifecycle"].is_terminal)

    def test_decode_pool_merges_metrics_with_worker_scoped_sequence_ids(self):
        coordinator = MultiDecodeCoordinator()
        engine = PDServingEngine(coordinator, prefill_batch_size=2)
        first = engine.add_request([1, 2], SamplingParams(max_tokens=2))
        second = engine.add_request([3, 4], SamplingParams(max_tokens=2))
        engine.step()

        metrics = engine.get_metrics()

        self.assertEqual([request["seq_id"] for request in metrics["requests"]], [
            first,
            second,
        ])
        self.assertEqual(
            set(metrics["summary"]["decode_workers"]),
            {"decode-0", "decode-1"},
        )

    def test_decode_pool_reset_metrics_resets_every_worker(self):
        coordinator = MultiDecodeCoordinator()
        engine = PDServingEngine(coordinator, prefill_batch_size=2)

        engine.reset_metrics()

        self.assertEqual(coordinator.reset_all_calls, 1)

    def test_decode_pool_cancellation_removes_targeted_pending_handoff(self):
        coordinator = MultiDecodeCoordinator()
        engine = PDServingEngine(coordinator, prefill_batch_size=2)
        first = engine.add_request([1, 2], SamplingParams(max_tokens=2))
        second = engine.add_request([3, 4], SamplingParams(max_tokens=2))
        for request_id in (first, second):
            request = engine._requests[request_id]
            request["lifecycle"].transition(RequestState.PREFILLING)
            request["lifecycle"].transition(RequestState.HANDOFF)

        def handoff_for(request_id, worker_id):
            return SimpleNamespace(
                request_id=request_id,
                descriptor=SimpleNamespace(
                    transport="shared_slot",
                    transfer_id=f"transfer-{request_id}",
                    target_worker=worker_id,
                ),
            )

        first_handoff = handoff_for(first, "decode-0")
        second_handoff = handoff_for(second, "decode-1")
        engine._pending_handoff_batch = {
            "handoffs": [first_handoff, second_handoff],
            "handoffs_by_worker": {
                "decode-0": [first_handoff],
                "decode-1": [second_handoff],
            },
            "meta": {"request_ids": [first, second], "batch_size": 2},
            "finished_at": 0.0,
            "prefill_rpc_timing": {},
        }

        self.assertTrue(engine.abort_request(first))

        self.assertEqual(engine._pending_transfer_acks, ["transfer-0"])
        self.assertEqual(
            engine._pending_handoff_batch["handoffs"], [second_handoff]
        )
        self.assertEqual(
            engine._pending_handoff_batch["handoffs_by_worker"],
            {"decode-1": [second_handoff]},
        )

    def test_decode_pool_queue_samples_are_scoped_per_worker(self):
        coordinator = MultiDecodeCoordinator()
        engine = PDServingEngine(coordinator, prefill_batch_size=2)
        first = engine.add_request([1, 2], SamplingParams(max_tokens=2))
        second = engine.add_request([3, 4], SamplingParams(max_tokens=2))
        engine._requests[first]["decode_worker_id"] = "decode-0"
        engine._requests[second]["decode_worker_id"] = "decode-1"
        engine._active_by_decode_seq = {
            ("decode-0", 0): first,
            ("decode-1", 0): second,
        }
        engine._awaiting_transfer_completions = {"transfer-0": first}

        engine._record_queue_sample()

        sample = engine._queue_samples[-1]
        self.assertEqual(
            sample["active_decode_by_worker"],
            {"decode-0": 1, "decode-1": 1},
        )
        self.assertEqual(
            sample["pending_kv_imports_by_worker"],
            {"decode-0": 1, "decode-1": 0},
        )

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

    def test_active_decode_combines_new_handoff_with_next_worker_step(self):
        coordinator = CombinedStepCoordinator()
        engine = PDServingEngine(
            coordinator,
            prefill_batch_size=1,
            enable_transport_overlap=True,
        )
        active_request_id = engine.add_request([1, 2], SamplingParams(max_tokens=2))
        handoff_request_id = engine.add_request([3, 4], SamplingParams(max_tokens=2))
        active_request = engine._requests[active_request_id]
        for state in (RequestState.PREFILLING, RequestState.HANDOFF, RequestState.DECODING):
            active_request["lifecycle"].transition(state)
        active_request["decode_seq_id"] = 100
        engine._active_by_decode_seq[100] = active_request_id

        handoff_request = engine._requests[handoff_request_id]
        for state in (RequestState.PREFILLING, RequestState.HANDOFF):
            handoff_request["lifecycle"].transition(state)
        handoff = SimpleNamespace(
            request_id=handoff_request_id,
            envelope=RequestEnvelope(
                handoff_request_id,
                (3, 4),
                2,
                1.0,
                True,
            ),
            descriptor=SimpleNamespace(transport="inline"),
            prefill_timing_ms={},
            telemetry={},
        )
        engine._pending_handoff_batch = {
            "handoffs": [handoff],
            "meta": {"request_ids": [handoff_request_id], "batch_size": 1},
            "finished_at": 0.0,
            "prefill_rpc_timing": {},
        }
        engine._pending.clear()

        outputs, num_tokens = engine.step()

        self.assertEqual(outputs, [(active_request_id, [200])])
        self.assertEqual(num_tokens, 1)
        self.assertEqual(coordinator.combined_handoffs, [[handoff]])
        detail = engine.get_metrics()["summary"]["pd"]["prefill_batches_detail"][0]
        self.assertIn("combined_step_roundtrip_ms", detail)
        self.assertEqual(engine._requests[handoff_request_id]["decode_seq_id"], 101)
        self.assertEqual(
            engine._requests[handoff_request_id]["lifecycle"].state,
            RequestState.DECODING,
        )

    def test_decode_idle_records_waiting_kv_import_from_worker_diagnostics(self):
        class KVWaitCoordinator(FakePDCoordinator):
            def decode_step(self):
                self.decode_calls += 1
                return {
                    "outputs": [],
                    "num_tokens": 0,
                    "last_step_events": {
                        "scheduled_seq_ids": [],
                        "waiting_queue_size": 0,
                        "running_queue_size": 0,
                    },
                    "step_diagnostics": {
                        "idle_reason": "waiting_kv_h2d",
                        "started_at": 1.0,
                        "finished_at": 1.25,
                    },
                }

        engine = PDServingEngine(
            KVWaitCoordinator(),
            prefill_batch_size=1,
            enable_latency_telemetry=True,
        )
        engine.add_request([1, 2], SamplingParams(max_tokens=2))
        engine.step()

        idle = engine.get_metrics()["summary"]["pd"]["decode_idle"]
        self.assertEqual(idle["counts"]["waiting_kv_h2d"], 1)
        self.assertEqual(idle["duration_ms"]["waiting_kv_h2d"], 250.0)

    def test_abort_handoff_request_removes_unsent_shared_transfer(self):
        engine = PDServingEngine(FakePDCoordinator(), prefill_batch_size=1)
        request_id = engine.add_request([1, 2], SamplingParams(max_tokens=2))
        request = engine._requests[request_id]
        for state in (RequestState.PREFILLING, RequestState.HANDOFF):
            request["lifecycle"].transition(state)
        handoff = SimpleNamespace(
            request_id=request_id,
            descriptor=SimpleNamespace(
                transport="shared_slot",
                transfer_id="transfer-0",
            ),
        )
        engine._pending_handoff_batch = {
            "handoffs": [handoff],
            "meta": {"request_ids": [request_id], "batch_size": 1},
            "finished_at": 0.0,
            "prefill_rpc_timing": {},
        }

        self.assertTrue(engine.abort_request(request_id))

        self.assertTrue(request["lifecycle"].is_terminal)
        self.assertIsNone(engine._pending_handoff_batch)
        self.assertEqual(engine._pending_transfer_acks, ["transfer-0"])

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
            def __init__(self):
                super().__init__()
                self.completed_transfers = []

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
                self.completed_transfers.extend(
                    {"transfer_id": handoff.descriptor.transfer_id}
                    for handoff in handoffs
                )
                return [
                    {
                        "seq_id": 100 + handoff.request_id,
                        "finished": False,
                        "output_token_ids": None,
                        "transfer_id": handoff.descriptor.transfer_id,
                    }
                    for handoff in handoffs
                ]

            def collect_completed_transfers(self, wait=False):
                completed = self.completed_transfers
                self.completed_transfers = []
                return completed

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
            [["transfer-1"], ["transfer-2"]],
        )

    def test_shared_transfer_ack_is_deferred_until_decode_reports_import_completion(self):
        class CompletionGatedCoordinator(FakePDCoordinator):
            def __init__(self):
                super().__init__()
                self.import_complete = False

            def prefill_batch(self, envelopes, release_transfer_ids=()):
                self.prefill_calls.append(list(envelopes))
                self.prefill_release_calls.append(list(release_transfer_ids))
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

            def collect_completed_transfers(self, wait=False):
                if self.import_complete:
                    self.import_complete = False
                    return [{"transfer_id": "transfer-0"}]
                return []

        coordinator = CompletionGatedCoordinator()
        engine = PDServingEngine(coordinator, prefill_batch_size=1)
        engine.add_request([1, 2], SamplingParams(max_tokens=2))

        engine.step()
        engine.step()

        self.assertEqual(coordinator.explicit_release_calls, [])
        self.assertFalse(engine.is_finished())

        coordinator.import_complete = True
        engine.step()

        self.assertEqual(coordinator.explicit_release_calls, [["transfer-0"]])
        self.assertTrue(engine.is_finished())

    def test_completed_transfer_propagates_device_overlap_telemetry(self):
        engine = PDServingEngine(
            FakePDCoordinator(),
            prefill_batch_size=1,
            enable_latency_telemetry=True,
        )
        request_id = engine.add_request([1, 2], SamplingParams(max_tokens=2))
        engine._awaiting_transfer_completions["transfer-0"] = request_id

        engine._record_completed_transfers([{
            "transfer_id": "transfer-0",
            "telemetry": {
                "copy_compute_overlap_ms": 0.8,
                "copy_compute_overlap_ratio": 0.8,
                "critical_path_reduction_gpu_ms": 0.8,
            },
            "kv_import_gpu_ms": 1.0,
        }])

        timeline = engine._requests[request_id]["timeline"]
        self.assertEqual(timeline["copy_compute_overlap_ms"], 0.8)
        self.assertEqual(timeline["copy_compute_overlap_ratio"], 0.8)
        self.assertEqual(timeline["critical_path_reduction_gpu_ms"], 0.8)

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

    def test_admission_failure_does_not_reuse_a_possibly_inflight_shared_transfer(self):
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

        self.assertEqual(coordinator.explicit_release_calls, [])
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
