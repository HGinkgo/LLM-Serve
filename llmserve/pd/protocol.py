"""Wire-stable request and KV handoff contracts for PD workers.

The protocol intentionally contains no torch or process-specific objects. A
worker owns its local Sequence and block table; only these portable contracts
cross the Prefill/Decode boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Mapping


class RequestState(str, Enum):
    QUEUED = "queued"
    PREFILLING = "prefilling"
    HANDOFF = "handoff"
    DECODING = "decoding"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"


class InvalidLifecycleTransition(RuntimeError):
    """Raised when a request attempts to skip or repeat a lifecycle stage."""


@dataclass(frozen=True, slots=True)
class RequestEnvelope:
    """Request data that can be sent to a Prefill Worker."""

    request_id: int
    prompt_token_ids: tuple[int, ...]
    max_tokens: int
    temperature: float
    ignore_eos: bool
    target_worker: str = "decode"

    def __post_init__(self):
        if not isinstance(self.request_id, int) or isinstance(self.request_id, bool):
            raise ValueError("request_id must be an integer")
        if self.request_id < 0:
            raise ValueError("request_id must be non-negative")
        token_ids = tuple(self.prompt_token_ids)
        if not token_ids or any(
            not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0
            for token_id in token_ids
        ):
            raise ValueError("prompt_token_ids must contain non-negative integers")
        object.__setattr__(self, "prompt_token_ids", token_ids)
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.temperature <= 1e-10:
            raise ValueError("temperature must be positive")
        if not isinstance(self.target_worker, str) or not self.target_worker:
            raise ValueError("target_worker must be a non-empty string")

    def to_payload(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prompt_token_ids": list(self.prompt_token_ids),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "ignore_eos": self.ignore_eos,
            "target_worker": self.target_worker,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RequestEnvelope":
        try:
            return cls(
                request_id=payload["request_id"],
                prompt_token_ids=tuple(payload["prompt_token_ids"]),
                max_tokens=payload["max_tokens"],
                temperature=payload["temperature"],
                ignore_eos=payload["ignore_eos"],
                target_worker=payload.get("target_worker", "decode"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid request envelope payload") from error


@dataclass(frozen=True, slots=True)
class KVTransferDescriptor:
    """Metadata for one Prefill -> Decode KV payload.

    The payload itself is owned by the transport implementation. The
    descriptor deliberately carries no CUDA tensor or worker-local block id.
    """

    request_id: int
    transfer_id: str
    num_tokens: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: str
    block_size: int
    payload_nbytes: int
    source_worker: str = "prefill"
    target_worker: str = "decode"
    layout_version: int = 1
    transport: str = "inline"
    slot_id: int | None = None
    slot_generation: int | None = None
    token_offset: int = 0

    def __post_init__(self):
        if not isinstance(self.request_id, int) or self.request_id < 0:
            raise ValueError("request_id must be non-negative")
        if not self.transfer_id:
            raise ValueError("transfer_id must not be empty")
        for name in (
            "num_tokens",
            "num_layers",
            "num_kv_heads",
            "head_dim",
            "block_size",
            "payload_nbytes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.dtype:
            raise ValueError("dtype must not be empty")
        if not self.source_worker or not self.target_worker:
            raise ValueError("worker names must not be empty")
        if self.source_worker == self.target_worker:
            raise ValueError("source and target workers must differ")
        if self.layout_version <= 0:
            raise ValueError("layout_version must be positive")
        if self.transport not in {"inline", "shared_slot"}:
            raise ValueError("unsupported KV transport")
        if self.transport == "shared_slot":
            if (
                not isinstance(self.slot_id, int)
                or self.slot_id < 0
                or not isinstance(self.slot_generation, int)
                or self.slot_generation <= 0
                or not isinstance(self.token_offset, int)
                or self.token_offset < 0
            ):
                raise ValueError("shared KV transport requires complete slot metadata")
            object.__setattr__(self, "layout_version", 2)
        elif self.slot_id is not None or self.slot_generation is not None:
            raise ValueError("inline KV transport cannot carry slot metadata")

    def to_payload(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "transfer_id": self.transfer_id,
            "num_tokens": self.num_tokens,
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "dtype": self.dtype,
            "block_size": self.block_size,
            "payload_nbytes": self.payload_nbytes,
            "source_worker": self.source_worker,
            "target_worker": self.target_worker,
            "layout_version": self.layout_version,
            "transport": self.transport,
            "slot_id": self.slot_id,
            "slot_generation": self.slot_generation,
            "token_offset": self.token_offset,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "KVTransferDescriptor":
        try:
            return cls(
                request_id=payload["request_id"],
                transfer_id=payload["transfer_id"],
                num_tokens=payload["num_tokens"],
                num_layers=payload["num_layers"],
                num_kv_heads=payload["num_kv_heads"],
                head_dim=payload["head_dim"],
                dtype=payload["dtype"],
                block_size=payload["block_size"],
                payload_nbytes=payload["payload_nbytes"],
                source_worker=payload.get("source_worker", "prefill"),
                target_worker=payload.get("target_worker", "decode"),
                layout_version=payload.get("layout_version", 1),
                transport=payload.get("transport", "inline"),
                slot_id=payload.get("slot_id"),
                slot_generation=payload.get("slot_generation"),
                token_offset=payload.get("token_offset", 0),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid KV transfer descriptor payload") from error


@dataclass(slots=True)
class RequestLifecycle:
    """Explicit request state machine shared by the PD coordinator."""

    request_id: int
    state: RequestState = RequestState.QUEUED
    terminal_reason: str | None = None

    _allowed_transitions: ClassVar[dict[RequestState, frozenset[RequestState]]] = {
        RequestState.QUEUED: frozenset({
            RequestState.PREFILLING,
            RequestState.FAILED,
            RequestState.CANCELLED,
        }),
        RequestState.PREFILLING: frozenset({
            RequestState.HANDOFF,
            RequestState.FAILED,
            RequestState.CANCELLED,
        }),
        RequestState.HANDOFF: frozenset({
            RequestState.DECODING,
            RequestState.FAILED,
            RequestState.CANCELLED,
        }),
        RequestState.DECODING: frozenset({
            RequestState.FINISHED,
            RequestState.FAILED,
            RequestState.CANCELLED,
        }),
        RequestState.FINISHED: frozenset(),
        RequestState.FAILED: frozenset(),
        RequestState.CANCELLED: frozenset(),
    }

    def __post_init__(self):
        if not isinstance(self.request_id, int) or self.request_id < 0:
            raise ValueError("request_id must be non-negative")

    @property
    def is_terminal(self) -> bool:
        return self.state in {
            RequestState.FINISHED,
            RequestState.FAILED,
            RequestState.CANCELLED,
        }

    def transition(self, next_state: RequestState, reason: str | None = None):
        if next_state == self.state:
            return
        if next_state not in self._allowed_transitions[self.state]:
            raise InvalidLifecycleTransition(
                f"request {self.request_id}: "
                f"cannot transition {self.state.value} -> {next_state.value}"
            )
        self.state = next_state
        if self.is_terminal and reason is not None:
            self.terminal_reason = reason

    def fail(self, reason: str):
        self.transition(RequestState.FAILED, reason=reason)

    def cancel(self, reason: str = "cancelled"):
        self.transition(RequestState.CANCELLED, reason=reason)
