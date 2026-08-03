"""Worker-local orchestration for the Prefill/Decode handoff path."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from uuid import uuid4

import torch

from llmserve.engine.scheduler import SchedulerOutput
from llmserve.pd.kv_transfer import export_logical_kv
from llmserve.pd.protocol import KVTransferDescriptor, RequestEnvelope
from llmserve.sampling_params import SamplingParams


@dataclass(slots=True)
class PrefillHandoff:
    envelope: RequestEnvelope
    first_token_id: int
    descriptor: KVTransferDescriptor
    kv_payload: torch.Tensor
    prefill_timing_ms: dict[str, float | int] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.first_token_id, int) or self.first_token_id < 0:
            raise ValueError("first_token_id must be a non-negative integer")
        if self.descriptor.request_id != self.envelope.request_id:
            raise ValueError("KV handoff request id does not match its envelope")
        if (
            self.descriptor.source_worker != "prefill"
            or self.descriptor.target_worker != "decode"
        ):
            raise ValueError("KV handoff has an invalid worker direction")
        if not isinstance(self.kv_payload, torch.Tensor):
            raise ValueError("KV handoff payload must be a tensor")
        expected_shape = (
            2,
            self.descriptor.num_layers,
            self.descriptor.num_tokens,
            self.descriptor.num_kv_heads,
            self.descriptor.head_dim,
        )
        if tuple(self.kv_payload.shape) != expected_shape:
            raise ValueError(
                "KV descriptor shape does not match its payload: "
                f"expected {expected_shape}, got {tuple(self.kv_payload.shape)}"
            )
        payload_dtype = str(self.kv_payload.dtype).replace("torch.", "")
        if payload_dtype != self.descriptor.dtype:
            raise ValueError("KV descriptor dtype does not match its payload")
        payload_nbytes = self.kv_payload.numel() * self.kv_payload.element_size()
        if payload_nbytes != self.descriptor.payload_nbytes:
            raise ValueError("KV descriptor byte size does not match its payload")

    @property
    def request_id(self) -> int:
        return self.envelope.request_id


def _to_host_payload(payload: torch.Tensor) -> torch.Tensor:
    if payload.device.type == "cpu":
        return payload.contiguous()
    host = torch.empty_like(
        payload,
        device="cpu",
        pin_memory=torch.cuda.is_available(),
    )
    host.copy_(payload, non_blocking=False)
    return host


class PrefillWorkerRuntime:
    """Run prompt prefill and detach its prompt KV from the local engine."""

    def __init__(self, engine):
        self.engine = engine

    def prefill(self, envelope: RequestEnvelope) -> PrefillHandoff:
        return self.prefill_batch([envelope])[0]

    def _build_handoff(
        self,
        envelope: RequestEnvelope,
        seq,
        first_token_id: int,
    ) -> PrefillHandoff:
        logical_tokens = seq.num_prompt_tokens
        payload = export_logical_kv(
            self.engine.model_runner.kv_cache,
            seq.block_table,
            logical_tokens,
        )
        payload = _to_host_payload(payload)
        descriptor = KVTransferDescriptor(
            request_id=envelope.request_id,
            transfer_id=f"{envelope.request_id}-{uuid4().hex}",
            num_tokens=logical_tokens,
            num_layers=payload.size(1),
            num_kv_heads=payload.size(3),
            head_dim=payload.size(4),
            dtype=str(payload.dtype).replace("torch.", ""),
            block_size=self.engine.model_runner.block_size,
            payload_nbytes=payload.numel() * payload.element_size(),
        )
        return PrefillHandoff(
            envelope=envelope,
            first_token_id=int(first_token_id),
            descriptor=descriptor,
            kv_payload=payload,
        )

    def prefill_batch(
        self,
        envelopes: list[RequestEnvelope],
    ) -> list[PrefillHandoff]:
        """Prefill several requests and return independent KV handoffs.

        Partial chunks are postprocessed back into the Prefill Worker. A
        sequence is removed as soon as its prompt KV and first token have been
        copied into a handoff, so completed requests never enter Prefill decode.
        """
        envelopes = list(envelopes)
        if not envelopes:
            raise ValueError("prefill_batch requires at least one request")
        request_ids = [envelope.request_id for envelope in envelopes]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("prefill_batch request ids must be unique")

        seq_by_id = {}
        envelope_by_seq_id = {}
        owned_seq_ids = set()
        handoffs_by_request_id = {}
        batch_started_at = perf_counter()
        model_forward_ms = 0.0
        kv_export_copy_ms = 0.0
        forward_calls = 0
        try:
            for envelope in envelopes:
                sampling_params = SamplingParams(
                    temperature=envelope.temperature,
                    max_tokens=envelope.max_tokens,
                    ignore_eos=envelope.ignore_eos,
                )
                seq_id = self.engine.add_request(
                    list(envelope.prompt_token_ids),
                    sampling_params,
                )
                seq = next(
                    seq
                    for seq in self.engine.scheduler.waiting
                    if seq.seq_id == seq_id
                )
                seq_by_id[seq_id] = seq
                envelope_by_seq_id[seq_id] = envelope
                owned_seq_ids.add(seq_id)

            while owned_seq_ids:
                scheduler_output = self.engine.scheduler.schedule()
                if not scheduler_output.prefill_seqs or scheduler_output.decode_seqs:
                    raise RuntimeError(
                        "PD Prefill Worker received a non-prefill scheduler output"
                    )
                forward_started_at = perf_counter()
                token_ids = self.engine.model_runner.call("run", scheduler_output)
                model_forward_ms += (perf_counter() - forward_started_at) * 1000
                forward_calls += 1
                if len(token_ids) != len(scheduler_output.scheduled_seqs):
                    raise RuntimeError(
                        "prefill worker returned an unexpected token count"
                    )

                partial_seqs = []
                partial_token_ids = []
                completed = []
                for seq, token_id in zip(
                    scheduler_output.prefill_seqs,
                    token_ids,
                ):
                    if seq.seq_id not in owned_seq_ids:
                        raise RuntimeError(
                            "Prefill Worker scheduled a sequence outside this batch"
                        )
                    cached_before = seq.num_cached_tokens
                    scheduled_tokens = seq.num_scheduled_tokens
                    if cached_before + scheduled_tokens < seq.num_tokens:
                        partial_seqs.append(seq)
                        partial_token_ids.append(token_id)
                        continue
                    export_started_at = perf_counter()
                    handoff = self._build_handoff(
                        envelope_by_seq_id[seq.seq_id],
                        seq,
                        token_id,
                    )
                    kv_export_copy_ms += (
                        perf_counter() - export_started_at
                    ) * 1000
                    completed.append((seq, handoff))

                if partial_seqs:
                    partial_output = SchedulerOutput(
                        partial_seqs,
                        partial_seqs,
                        [],
                        sum(seq.num_scheduled_tokens for seq in partial_seqs),
                    )
                    self.engine.scheduler.postprocess(
                        partial_output,
                        partial_token_ids,
                    )

                for seq, handoff in completed:
                    handoffs_by_request_id[handoff.request_id] = handoff
                    self.engine.scheduler.remove_sequence(seq)
                    owned_seq_ids.remove(seq.seq_id)

            batch_timing = {
                "worker_total_ms": (perf_counter() - batch_started_at) * 1000,
                "model_forward_ms": model_forward_ms,
                "kv_export_copy_ms": kv_export_copy_ms,
                "forward_calls": forward_calls,
            }
            for handoff in handoffs_by_request_id.values():
                handoff.prefill_timing_ms = dict(batch_timing)
            return [handoffs_by_request_id[request_id] for request_id in request_ids]
        except Exception:
            for seq_id in list(owned_seq_ids):
                self.engine.scheduler.remove_sequence(seq_by_id[seq_id])
            raise


class DecodeWorkerRuntime:
    """Admit a Prefill handoff into the Decode Worker engine."""

    def __init__(self, engine):
        self.engine = engine

    def admit_batch(self, handoffs: list[PrefillHandoff]) -> list[dict]:
        handoffs = list(handoffs)
        if not handoffs:
            raise ValueError("admit_batch requires at least one handoff")
        request_ids = [handoff.request_id for handoff in handoffs]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("admit_batch request ids must be unique")
        return [self.admit(handoff) for handoff in handoffs]

    def admit(self, handoff: PrefillHandoff):
        if handoff.descriptor.request_id != handoff.envelope.request_id:
            raise ValueError("KV handoff request id does not match its envelope")
        if handoff.descriptor.block_size != self.engine.model_runner.block_size:
            raise ValueError("Prefill and Decode Worker block sizes do not match")
        sampling_params = SamplingParams(
            temperature=handoff.envelope.temperature,
            max_tokens=handoff.envelope.max_tokens,
            ignore_eos=handoff.envelope.ignore_eos,
        )
        seq_id = self.engine.add_prefilled_request(
            list(handoff.envelope.prompt_token_ids),
            handoff.first_token_id,
            sampling_params,
            handoff.kv_payload,
        )
        finished = (
            handoff.envelope.max_tokens <= 1
            or (
                not handoff.envelope.ignore_eos
                and handoff.first_token_id == self.engine.scheduler.eos
            )
        )
        return {
            "seq_id": seq_id,
            "finished": finished,
            "output_token_ids": [handoff.first_token_id] if finished else None,
        }
