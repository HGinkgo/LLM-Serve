import unittest
from types import SimpleNamespace

import torch

from llmserve.engine.scheduler import SchedulerOutput
from llmserve.engine.sequence import Sequence
from llmserve.pd.protocol import KVTransferDescriptor, RequestEnvelope
from llmserve.pd.runtime import (
    DecodeWorkerRuntime,
    PrefillHandoff,
    PrefillWorkerRuntime,
)
from llmserve.pd.shared_slots import SharedKVSlotPool, SharedKVSlotReader
from llmserve.sampling_params import SamplingParams


class FakePrefillScheduler:

    def __init__(self, seq):
        self.seq = seq
        self.waiting = [seq]
        self.removed = []

    def schedule(self):
        self.seq.num_scheduled_tokens = len(self.seq)
        self.seq.block_table = [0]
        return SchedulerOutput([self.seq], [self.seq], [], len(self.seq))

    def remove_sequence(self, seq):
        self.removed.append(seq)


class FakePrefillEngine:

    def __init__(self, prompt):
        self.seq = Sequence(prompt, SamplingParams(temperature=1.0, max_tokens=4))
        self.scheduler = FakePrefillScheduler(self.seq)
        self.model_runner = SimpleNamespace(
            kv_cache=torch.full((2, 1, 1, 4, 1, 2), 5.0),
            block_size=4,
        )

    def add_request(self, prompt, sampling_params):
        self.seq = Sequence(prompt, sampling_params)
        self.scheduler.seq = self.seq
        self.scheduler.waiting = [self.seq]
        return self.seq.seq_id

    def call(self, method_name, scheduler_output):
        raise AssertionError(method_name)


class FakeDecodeEngine:

    def __init__(self):
        self.calls = []
        self.model_runner = SimpleNamespace(block_size=4)
        self.scheduler = SimpleNamespace(eos=99)

    def add_prefilled_request(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return 123

    def abort_request(self, seq_id):
        self.calls.append(("abort_request", seq_id))
        return True


class FakeBatchPrefillScheduler:

    def __init__(self):
        self.seqs = []
        self.waiting = self.seqs
        self.removed = []

    def schedule(self):
        scheduled = [
            seq for seq in self.seqs if seq.num_cached_tokens < seq.num_tokens
        ]
        for block_id, seq in enumerate(scheduled):
            seq.block_table = [block_id]
            seq.num_scheduled_tokens = seq.num_tokens - seq.num_cached_tokens
        return SchedulerOutput(
            scheduled,
            scheduled,
            [],
            sum(seq.num_scheduled_tokens for seq in scheduled),
        )

    def remove_sequence(self, seq):
        self.removed.append(seq)
        self.seqs.remove(seq)


class FakeBatchPrefillEngine:

    def __init__(self):
        self.scheduler = FakeBatchPrefillScheduler()
        self.model_runner = SimpleNamespace(
            kv_cache=torch.zeros(2, 1, 2, 4, 1, 2),
            block_size=4,
        )
        self.model_runner.call = lambda method_name, scheduler_output: [101, 102]

    def add_request(self, prompt, sampling_params):
        seq = Sequence(prompt, sampling_params)
        self.scheduler.seqs.append(seq)
        return seq.seq_id


class FakeChunkedPrefillScheduler:

    def __init__(self):
        self.seqs = []
        self.waiting = self.seqs
        self.removed = []
        self.postprocess_calls = 0

    def schedule(self):
        scheduled = [
            seq for seq in self.seqs if seq.seq_id not in self.removed
        ]
        for seq in scheduled:
            seq.block_table = [0, 1]
            seq.num_scheduled_tokens = min(2, seq.num_tokens - seq.num_cached_tokens)
        return SchedulerOutput(
            scheduled,
            scheduled,
            [],
            sum(seq.num_scheduled_tokens for seq in scheduled),
        )

    def postprocess(self, output, token_ids):
        self.postprocess_calls += 1
        for seq in output.prefill_seqs:
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

    def remove_sequence(self, seq):
        self.removed.append(seq.seq_id)
        self.seqs.remove(seq)


class FakeChunkedPrefillEngine:

    def __init__(self):
        self.scheduler = FakeChunkedPrefillScheduler()
        self.model_runner = SimpleNamespace(
            kv_cache=torch.zeros(2, 1, 2, 4, 1, 2),
            block_size=4,
        )
        self.call_count = 0
        self.model_runner.call = self.call

    def add_request(self, prompt, sampling_params):
        seq = Sequence(prompt, sampling_params)
        self.scheduler.seqs.append(seq)
        return seq.seq_id

    def call(self, method_name, scheduler_output):
        self.call_count += 1
        return [100 + self.call_count]

class TestPDRuntime(unittest.TestCase):

    @staticmethod
    def make_descriptor(payload, *, request_id=7, block_size=4, num_tokens=None):
        return KVTransferDescriptor(
            request_id=request_id,
            transfer_id="transfer-1",
            num_tokens=payload.size(2) if num_tokens is None else num_tokens,
            num_layers=payload.size(1),
            num_kv_heads=payload.size(3),
            head_dim=payload.size(4),
            dtype=str(payload.dtype).replace("torch.", ""),
            block_size=block_size,
            payload_nbytes=payload.numel() * payload.element_size(),
        )

    def test_prefill_runtime_exports_complete_prompt_and_first_token(self):
        prompt = [1, 2, 3, 4]
        engine = FakePrefillEngine(prompt)
        engine.model_runner.call = lambda method_name, output: [77]
        envelope = RequestEnvelope(7, tuple(prompt), 4, 1.0, True)

        result = PrefillWorkerRuntime(engine).prefill(envelope)

        self.assertEqual(result.request_id, 7)
        self.assertEqual(result.first_token_id, 77)
        self.assertEqual(result.descriptor.num_tokens, len(prompt))
        self.assertEqual(result.descriptor.source_worker, "prefill")
        self.assertEqual(result.kv_payload.shape, (2, 1, 4, 1, 2))
        self.assertEqual(engine.scheduler.removed, [engine.seq])

    def test_prefill_runtime_batches_handoffs_and_releases_each_sequence(self):
        engine = FakeBatchPrefillEngine()
        envelopes = [
            RequestEnvelope(7, (1, 2, 3, 4), 4, 1.0, True),
            RequestEnvelope(8, (5, 6), 4, 1.0, True),
        ]
        engine.model_runner.kv_cache[:, :, 0] = 7.0
        engine.model_runner.kv_cache[:, :, 1] = 8.0

        handoffs = PrefillWorkerRuntime(engine).prefill_batch(envelopes)

        self.assertEqual([handoff.request_id for handoff in handoffs], [7, 8])
        self.assertEqual([handoff.first_token_id for handoff in handoffs], [101, 102])
        self.assertEqual([handoff.descriptor.num_tokens for handoff in handoffs], [4, 2])
        self.assertEqual(
            [float(handoff.kv_payload[0, 0, 0, 0, 0]) for handoff in handoffs],
            [7.0, 8.0],
        )
        self.assertEqual(len(engine.scheduler.removed), 2)

    def test_prefill_runtime_reports_worker_timing_breakdown(self):
        engine = FakeBatchPrefillEngine()
        envelopes = [
            RequestEnvelope(7, (1, 2, 3, 4), 4, 1.0, True),
            RequestEnvelope(8, (5, 6), 4, 1.0, True),
        ]

        handoffs = PrefillWorkerRuntime(engine).prefill_batch(envelopes)

        timing = handoffs[0].prefill_timing_ms
        self.assertIn("worker_total_ms", timing)
        self.assertIn("model_forward_ms", timing)
        self.assertIn("kv_export_copy_ms", timing)
        self.assertEqual(timing["forward_calls"], 1)
        self.assertGreaterEqual(timing["worker_total_ms"], 0.0)
        self.assertGreaterEqual(timing["model_forward_ms"], 0.0)
        self.assertGreaterEqual(timing["kv_export_copy_ms"], 0.0)

    def test_shared_slot_handoff_carries_only_descriptors_and_decode_reads_slices(self):
        prefill_engine = FakeBatchPrefillEngine()
        prefill_engine.config = SimpleNamespace(enable_latency_telemetry=True)
        prefill_engine.model_runner.call = (
            lambda method_name, output=None: (
                {"model_forward_gpu_ms": 0.0}
                if method_name == "get_last_run_timing"
                else [101, 102]
            )
        )
        prefill_engine.model_runner.kv_cache[:, :, 0] = 7.0
        prefill_engine.model_runner.kv_cache[:, :, 1] = 8.0
        pool = SharedKVSlotPool.create(
            slot_count=2,
            capacity_tokens=8,
            num_layers=1,
            num_kv_heads=1,
            head_dim=2,
            dtype=torch.float32,
            register_cuda=False,
        )
        envelopes = [
            RequestEnvelope(7, (1, 2, 3, 4), 4, 1.0, True),
            RequestEnvelope(8, (5, 6), 4, 1.0, True),
        ]

        handoffs = PrefillWorkerRuntime(prefill_engine, slot_pool=pool).prefill_batch(
            envelopes
        )

        self.assertTrue(all(handoff.kv_payload is None for handoff in handoffs))
        self.assertEqual(
            [handoff.descriptor.transport for handoff in handoffs],
            ["shared_slot", "shared_slot"],
        )
        self.assertEqual(
            [handoff.descriptor.token_offset for handoff in handoffs],
            [0, 4],
        )
        source_timeline = handoffs[0].telemetry
        for name in (
            "t_slot_acquired",
            "t_kv_export_started",
            "t_kv_export_finished",
            "t_slot_ready",
        ):
            self.assertIn(name, source_timeline)
        self.assertIn("slot_observability", source_timeline)

        decode_engine = FakeDecodeEngine()
        decode_engine.config = SimpleNamespace(enable_latency_telemetry=True)
        reader = SharedKVSlotReader(pool.handle, register_cuda=False)
        admissions = DecodeWorkerRuntime(
            decode_engine,
            slot_reader=reader,
        ).admit_batch(handoffs)

        self.assertEqual(
            [admission["transfer_id"] for admission in admissions],
            [handoff.descriptor.transfer_id for handoff in handoffs],
        )
        self.assertTrue(all(
            admission["telemetry"]["t_slot_consuming"] is not None
            for admission in admissions
        ))
        self.assertEqual(
            [float(call[0][3][0, 0, 0, 0, 0]) for call in decode_engine.calls],
            [7.0, 8.0],
        )

        PrefillWorkerRuntime(prefill_engine, slot_pool=pool).release_transfers(
            [handoff.descriptor.transfer_id for handoff in handoffs]
        )
        self.assertEqual(pool.stats()["free_slots"], 2)

    def test_prefill_runtime_completes_partial_chunks_before_handoff(self):
        engine = FakeChunkedPrefillEngine()
        envelope = RequestEnvelope(7, (1, 2, 3, 4, 5), 4, 1.0, True)

        handoff = PrefillWorkerRuntime(engine).prefill(envelope)

        self.assertEqual(engine.call_count, 3)
        self.assertEqual(engine.scheduler.postprocess_calls, 2)
        self.assertEqual(handoff.first_token_id, 103)
        self.assertEqual(handoff.descriptor.num_tokens, 5)
        self.assertEqual(len(engine.scheduler.removed), 1)

    def test_decode_runtime_forwards_handoff_to_engine_bridge(self):
        envelope = RequestEnvelope(7, (1, 2, 3, 4), 4, 1.0, True)
        payload = torch.zeros(2, 1, 4, 1, 2)
        descriptor = self.make_descriptor(payload)
        handoff = PrefillHandoff(
            envelope=envelope,
            first_token_id=77,
            descriptor=descriptor,
            kv_payload=payload,
        )
        engine = FakeDecodeEngine()

        admission = DecodeWorkerRuntime(engine).admit(handoff)

        self.assertEqual(
            admission,
            {"seq_id": 123, "finished": False, "output_token_ids": None},
        )
        args, kwargs = engine.calls[0]
        self.assertEqual(args[0], [1, 2, 3, 4])
        self.assertEqual(args[1], 77)
        self.assertEqual(args[2].max_tokens, 4)
        self.assertTrue(torch.equal(args[3], handoff.kv_payload))
        self.assertEqual(kwargs, {})

    def test_decode_runtime_records_descriptor_and_import_enqueue_boundaries(self):
        envelope = RequestEnvelope(7, (1, 2, 3, 4), 4, 1.0, True)
        payload = torch.zeros(2, 1, 4, 1, 2)
        handoff = PrefillHandoff(
            envelope=envelope,
            first_token_id=77,
            descriptor=self.make_descriptor(payload),
            kv_payload=payload,
        )
        engine = FakeDecodeEngine()
        engine.config = SimpleNamespace(enable_latency_telemetry=True)

        admission = DecodeWorkerRuntime(engine).admit(handoff)

        telemetry = admission["telemetry"]
        self.assertLessEqual(
            telemetry["t_decode_descriptor_received"],
            telemetry["t_decode_admission_started"],
        )
        self.assertLessEqual(
            telemetry["t_decode_admission_started"],
            telemetry["t_decode_admission_returned"],
        )
        self.assertEqual(
            telemetry["writers"]["t_decode_descriptor_received"],
            "decode_worker.runtime",
        )
        self.assertEqual(
            telemetry["writers"]["t_decode_admission_returned"],
            "decode_worker.runtime",
        )

    def test_handoff_rejects_payload_that_disagrees_with_descriptor(self):
        envelope = RequestEnvelope(7, (1, 2, 3, 4), 4, 1.0, True)
        payload = torch.zeros(2, 1, 4, 1, 2)
        descriptor = self.make_descriptor(payload, num_tokens=3)

        with self.assertRaisesRegex(ValueError, "descriptor shape"):
            PrefillHandoff(
                envelope=envelope,
                first_token_id=77,
                descriptor=descriptor,
                kv_payload=payload,
            )

    def test_decode_runtime_rejects_worker_block_size_mismatch(self):
        envelope = RequestEnvelope(7, (1, 2, 3, 4), 4, 1.0, True)
        payload = torch.zeros(2, 1, 4, 1, 2)
        handoff = PrefillHandoff(
            envelope=envelope,
            first_token_id=77,
            descriptor=self.make_descriptor(payload, block_size=8),
            kv_payload=payload,
        )

        with self.assertRaisesRegex(ValueError, "block size"):
            DecodeWorkerRuntime(FakeDecodeEngine()).admit(handoff)

    def test_decode_runtime_reports_first_token_eos_completion(self):
        envelope = RequestEnvelope(7, (1, 2, 3, 4), 4, 1.0, False)
        payload = torch.zeros(2, 1, 4, 1, 2)
        handoff = PrefillHandoff(
            envelope=envelope,
            first_token_id=99,
            descriptor=self.make_descriptor(payload),
            kv_payload=payload,
        )

        admission = DecodeWorkerRuntime(FakeDecodeEngine()).admit(handoff)

        self.assertEqual(
            admission,
            {"seq_id": 123, "finished": True, "output_token_ids": [99]},
        )

    def test_decode_runtime_admits_multiple_handoffs_before_decode_step(self):
        payload = torch.zeros(2, 1, 4, 1, 2)
        handoffs = []
        for request_id, first_token_id in ((7, 77), (8, 88)):
            envelope = RequestEnvelope(
                request_id,
                (1, 2, 3, 4),
                4,
                1.0,
                True,
            )
            handoffs.append(
                PrefillHandoff(
                    envelope=envelope,
                    first_token_id=first_token_id,
                    descriptor=self.make_descriptor(payload, request_id=request_id),
                    kv_payload=payload,
                )
            )

        admissions = DecodeWorkerRuntime(FakeDecodeEngine()).admit_batch(handoffs)

        self.assertEqual(len(admissions), 2)
        self.assertEqual([admission["seq_id"] for admission in admissions], [123, 123])

    def test_decode_runtime_aborts_engine_request(self):
        engine = FakeDecodeEngine()
        runtime = DecodeWorkerRuntime(engine)

        self.assertTrue(runtime.abort_request(123))
        self.assertEqual(engine.calls, [("abort_request", 123)])


if __name__ == "__main__":
    unittest.main()
