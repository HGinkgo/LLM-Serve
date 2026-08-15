"""Logical KV packing for the CPU-relay PD handoff."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


def _validate_cache(kv_cache: torch.Tensor):
    if not isinstance(kv_cache, torch.Tensor) or kv_cache.ndim != 6:
        raise ValueError(
            "KV cache must have shape [2, layers, blocks, block_size, heads, dim]"
        )
    if kv_cache.size(0) != 2:
        raise ValueError("KV cache first dimension must contain K and V")
    if any(size <= 0 for size in kv_cache.shape[1:]):
        raise ValueError("KV cache dimensions must be positive")


def _validate_mapping(
    kv_cache: torch.Tensor,
    block_table: Sequence[int],
    num_tokens: int,
):
    _validate_cache(kv_cache)
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    block_size = kv_cache.size(3)
    required_blocks = (num_tokens + block_size - 1) // block_size
    if len(block_table) < required_blocks:
        raise ValueError("block table cannot cover the requested token range")
    for block_id in block_table[:required_blocks]:
        if not isinstance(block_id, int) or block_id < 0 or block_id >= kv_cache.size(2):
            raise ValueError("block table contains an invalid physical block id")


@dataclass(slots=True)
class KVImportCompletion:
    """Target-GPU completion boundary for one asynchronous KV import."""

    device: torch.device
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event

    def is_complete(self) -> bool:
        return bool(self.end_event.query())

    def wait_on_current_stream(self):
        torch.cuda.current_stream(self.device).wait_event(self.end_event)

    def synchronize(self):
        self.end_event.synchronize()

    def elapsed_ms(self) -> float:
        if not self.is_complete():
            raise RuntimeError("KV import timing requires a completed CUDA Event")
        return float(self.start_event.elapsed_time(self.end_event))


def export_logical_kv(
    kv_cache: torch.Tensor,
    block_table: Sequence[int],
    num_tokens: int,
) -> torch.Tensor:
    """Pack logical token order from paged KV storage into a contiguous tensor."""

    _validate_mapping(kv_cache, block_table, num_tokens)
    block_size = kv_cache.size(3)
    payload = torch.empty(
        (2, kv_cache.size(1), num_tokens, kv_cache.size(4), kv_cache.size(5)),
        dtype=kv_cache.dtype,
        device=kv_cache.device,
    )
    for start in range(0, num_tokens, block_size):
        count = min(block_size, num_tokens - start)
        block_id = block_table[start // block_size]
        payload[:, :, start:start + count] = kv_cache[
            :, :, block_id, :count
        ]
    return payload


def import_logical_kv(
    kv_cache: torch.Tensor,
    block_table: Sequence[int],
    payload: torch.Tensor,
    *,
    stream: torch.cuda.Stream | None = None,
    non_blocking: bool | None = None,
) -> KVImportCompletion | None:
    """Write contiguous logical KV into the target paged KV storage."""

    if not isinstance(payload, torch.Tensor) or payload.ndim != 5:
        raise ValueError("KV payload must have shape [2, layers, tokens, heads, dim]")
    _validate_mapping(kv_cache, block_table, payload.size(2))
    expected_shape = (
        2,
        kv_cache.size(1),
        payload.size(2),
        kv_cache.size(4),
        kv_cache.size(5),
    )
    if tuple(payload.shape) != expected_shape:
        raise ValueError(
            f"KV payload shape {tuple(payload.shape)} does not match {expected_shape}"
        )
    if payload.dtype != kv_cache.dtype:
        raise ValueError("KV payload dtype must match target KV cache dtype")

    if stream is not None and kv_cache.device.type != "cuda":
        raise ValueError("an asynchronous KV import requires a CUDA KV cache")

    copy_non_blocking = (
        payload.is_pinned() if non_blocking is None else bool(non_blocking)
    )

    def copy_and_scatter():
        target_payload = payload
        if target_payload.device != kv_cache.device:
            target_payload = target_payload.to(
                device=kv_cache.device,
                non_blocking=copy_non_blocking,
            )
        block_size = kv_cache.size(3)
        for start in range(0, target_payload.size(2), block_size):
            count = min(block_size, target_payload.size(2) - start)
            block_id = block_table[start // block_size]
            kv_cache[:, :, block_id, :count] = target_payload[
                :, :, start:start + count
            ]

    if stream is None:
        copy_and_scatter()
        return None

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start_event.record(stream)
        copy_and_scatter()
        end_event.record(stream)
    return KVImportCompletion(
        device=kv_cache.device,
        start_event=start_event,
        end_event=end_event,
    )
