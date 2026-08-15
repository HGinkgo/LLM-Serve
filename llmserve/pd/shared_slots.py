"""Reusable shared-memory slots for Prefill/Decode KV transfer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from time import perf_counter
from typing import Callable

import torch

from llmserve.pd.observability import collect_process_numa_observability


class KVSlotState(str, Enum):
    FREE = "free"
    FILLING = "filling"
    READY = "ready"
    CONSUMING = "consuming"


_STATE_CODES = {
    KVSlotState.FREE: 0,
    KVSlotState.FILLING: 1,
    KVSlotState.READY: 2,
    KVSlotState.CONSUMING: 3,
}


class KVSlotPoolExhausted(RuntimeError):
    """Raised when every bounded transport slot is still in use."""


@dataclass(frozen=True, slots=True)
class KVSlotLease:
    slot_id: int
    generation: int
    num_tokens: int


@dataclass(frozen=True, slots=True)
class SharedKVSlotHandle:
    backing: torch.Tensor
    states: torch.Tensor
    generations: torch.Tensor
    slot_count: int
    capacity_tokens: int
    num_layers: int
    num_kv_heads: int
    head_dim: int

    @property
    def elements_per_token(self) -> int:
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim

    def __post_init__(self):
        expected_shape = (
            self.slot_count,
            self.capacity_tokens * self.elements_per_token,
        )
        if tuple(self.backing.shape) != expected_shape:
            raise ValueError("shared KV backing shape does not match its handle")
        if self.backing.device.type != "cpu" or not self.backing.is_shared():
            raise ValueError("shared KV backing must use CPU shared memory")
        for name, tensor in (
            ("states", self.states),
            ("generations", self.generations),
        ):
            if (
                tuple(tensor.shape) != (self.slot_count,)
                or tensor.device.type != "cpu"
                or not tensor.is_shared()
            ):
                raise ValueError(f"shared KV {name} metadata is invalid")


def _register_cuda_host_memory(tensor: torch.Tensor) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA host registration requires an available CUDA device")
    runtime = torch.cuda.cudart()
    error = runtime.cudaHostRegister(
        tensor.data_ptr(),
        tensor.numel() * tensor.element_size(),
        1,
    )
    if error == runtime.cudaError.success:
        return True
    if "AlreadyRegistered" in str(error):
        return False
    raise RuntimeError(f"cudaHostRegister failed for shared KV slots: {error}")


class SharedKVSlotReader:
    """Decode-side view over a shared KV slot backing."""

    def __init__(self, handle: SharedKVSlotHandle, *, register_cuda: bool = True):
        self.handle = handle
        self._registered = (
            _register_cuda_host_memory(handle.backing) if register_cuda else False
        )

    def read(
        self,
        *,
        slot_id: int,
        generation: int,
        token_offset: int,
        num_tokens: int,
    ) -> torch.Tensor:
        if not isinstance(slot_id, int) or not 0 <= slot_id < self.handle.slot_count:
            raise ValueError("shared KV slot id is out of range")
        if int(self.handle.generations[slot_id]) != generation:
            raise ValueError("shared KV descriptor has a stale generation")
        state_code = int(self.handle.states[slot_id])
        if state_code not in {
            _STATE_CODES[KVSlotState.READY],
            _STATE_CODES[KVSlotState.CONSUMING],
        }:
            raise ValueError("shared KV slot is not ready for consumption")
        if token_offset < 0 or num_tokens <= 0:
            raise ValueError("shared KV token range must be positive")
        token_end = token_offset + num_tokens
        if token_end > self.handle.capacity_tokens:
            raise ValueError("shared KV token range exceeds slot capacity")
        element_offset = token_offset * self.handle.elements_per_token
        num_elements = num_tokens * self.handle.elements_per_token
        result = self.handle.backing[
            slot_id,
            element_offset:element_offset + num_elements,
        ].view(
            2,
            self.handle.num_layers,
            num_tokens,
            self.handle.num_kv_heads,
            self.handle.head_dim,
        )
        self.handle.states[slot_id] = _STATE_CODES[KVSlotState.CONSUMING]
        return result

    def read_descriptor(self, descriptor) -> torch.Tensor:
        expected_dtype = str(self.handle.backing.dtype).replace("torch.", "")
        expected_nbytes = (
            descriptor.num_tokens
            * self.handle.elements_per_token
            * self.handle.backing.element_size()
        )
        if (
            descriptor.transport != "shared_slot"
            or descriptor.num_layers != self.handle.num_layers
            or descriptor.num_kv_heads != self.handle.num_kv_heads
            or descriptor.head_dim != self.handle.head_dim
            or descriptor.dtype != expected_dtype
            or descriptor.payload_nbytes != expected_nbytes
        ):
            raise ValueError("shared KV descriptor geometry does not match its pool")
        return self.read(
            slot_id=descriptor.slot_id,
            generation=descriptor.slot_generation,
            token_offset=descriptor.token_offset,
            num_tokens=descriptor.num_tokens,
        )

    def close(self):
        if self._registered:
            error = torch.cuda.cudart().cudaHostUnregister(
                self.handle.backing.data_ptr()
            )
            if error != torch.cuda.cudart().cudaError.success:
                raise RuntimeError(
                    f"cudaHostUnregister failed for shared KV slots: {error}"
                )
            self._registered = False


class SharedKVSlotPool(SharedKVSlotReader):
    """Prefill-side owner of reusable shared KV slots."""

    @classmethod
    def create(
        cls,
        *,
        slot_count: int,
        capacity_tokens: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        register_cuda: bool = True,
        clock: Callable[[], float] = perf_counter,
        environment_provider: Callable[..., dict] = collect_process_numa_observability,
    ) -> "SharedKVSlotPool":
        for name, value in (
            ("slot_count", slot_count),
            ("capacity_tokens", capacity_tokens),
            ("num_layers", num_layers),
            ("num_kv_heads", num_kv_heads),
            ("head_dim", head_dim),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        backing = torch.empty(
            (
                slot_count,
                capacity_tokens * 2 * num_layers * num_kv_heads * head_dim,
            ),
            dtype=dtype,
            device="cpu",
        ).share_memory_()
        states = torch.full(
            (slot_count,),
            _STATE_CODES[KVSlotState.FREE],
            dtype=torch.uint8,
        ).share_memory_()
        generations = torch.zeros(slot_count, dtype=torch.int64).share_memory_()
        handle = SharedKVSlotHandle(
            backing=backing,
            states=states,
            generations=generations,
            slot_count=slot_count,
            capacity_tokens=capacity_tokens,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        environment = environment_provider(
            shared_memory_address=backing.data_ptr(),
        )
        return cls(
            handle,
            register_cuda=register_cuda,
            clock=clock,
            environment=environment,
        )

    def __init__(
        self,
        handle: SharedKVSlotHandle,
        *,
        register_cuda: bool = True,
        clock: Callable[[], float] = perf_counter,
        environment: dict | None = None,
    ):
        super().__init__(handle, register_cuda=register_cuda)
        self._clock = clock
        self._environment = dict(environment or {})
        self._states = [KVSlotState.FREE] * handle.slot_count
        self._generations = [0] * handle.slot_count
        self._leases: dict[int, KVSlotLease] = {}
        self._pending_transfers: dict[str, KVSlotLease] = {}
        self._consuming_transfers: set[str] = set()
        self._events: list[dict] = []
        self._state_started_at: dict[int, tuple[KVSlotState, float]] = {}
        self._state_durations_seconds = {
            state: 0.0 for state in KVSlotState
        }
        self._acquire_exhaustions = 0

    def _record_state(
        self,
        lease: KVSlotLease,
        state: KVSlotState,
        *,
        writer: str,
    ):
        now = self._clock()
        previous = self._state_started_at.get(lease.slot_id)
        if previous is not None:
            previous_state, started_at = previous
            self._state_durations_seconds[previous_state] += now - started_at
        self._state_started_at[lease.slot_id] = (state, now)
        self._events.append({
            "slot_id": lease.slot_id,
            "generation": lease.generation,
            "state": state.value,
            "at": now,
            "writer": writer,
        })

    def acquire(self, num_tokens: int) -> KVSlotLease:
        if not isinstance(num_tokens, int) or num_tokens <= 0:
            raise ValueError("shared KV lease token count must be positive")
        if num_tokens > self.handle.capacity_tokens:
            raise ValueError("shared KV lease exceeds slot capacity")
        try:
            slot_id = self._states.index(KVSlotState.FREE)
        except ValueError as error:
            self._acquire_exhaustions += 1
            raise KVSlotPoolExhausted("no reusable shared KV slot is available") from error
        self._generations[slot_id] += 1
        lease = KVSlotLease(slot_id, self._generations[slot_id], num_tokens)
        self._states[slot_id] = KVSlotState.FILLING
        self.handle.states[slot_id] = _STATE_CODES[KVSlotState.FILLING]
        self.handle.generations[slot_id] = lease.generation
        self._leases[slot_id] = lease
        self._record_state(
            lease,
            KVSlotState.FILLING,
            writer="prefill_worker.slot_pool",
        )
        return lease

    def _validate_lease(self, lease: KVSlotLease, state: KVSlotState):
        active = self._leases.get(lease.slot_id)
        if active != lease or self._states[lease.slot_id] != state:
            raise ValueError("shared KV lease is stale or in the wrong state")

    def writable_view(
        self,
        lease: KVSlotLease,
        *,
        token_offset: int,
        num_tokens: int,
    ) -> torch.Tensor:
        self._validate_lease(lease, KVSlotState.FILLING)
        if token_offset < 0 or num_tokens <= 0:
            raise ValueError("shared KV token range must be positive")
        if token_offset + num_tokens > lease.num_tokens:
            raise ValueError("shared KV token range exceeds its lease")
        element_offset = token_offset * self.handle.elements_per_token
        num_elements = num_tokens * self.handle.elements_per_token
        return self.handle.backing[
            lease.slot_id,
            element_offset:element_offset + num_elements,
        ].view(
            2,
            self.handle.num_layers,
            num_tokens,
            self.handle.num_kv_heads,
            self.handle.head_dim,
        )

    def mark_ready(self, lease: KVSlotLease, transfer_ids: set[str]):
        self._validate_lease(lease, KVSlotState.FILLING)
        transfer_ids = set(transfer_ids)
        if not transfer_ids or any(not transfer_id for transfer_id in transfer_ids):
            raise ValueError("shared KV slot requires non-empty transfer ids")
        duplicates = transfer_ids.intersection(self._pending_transfers)
        if duplicates:
            raise ValueError("shared KV transfer id is already active")
        self._states[lease.slot_id] = KVSlotState.READY
        self.handle.states[lease.slot_id] = _STATE_CODES[KVSlotState.READY]
        self._record_state(
            lease,
            KVSlotState.READY,
            writer="prefill_worker.slot_pool",
        )
        for transfer_id in transfer_ids:
            self._pending_transfers[transfer_id] = lease

    def mark_consuming(self, transfer_ids: set[str]):
        transfer_ids = set(transfer_ids)
        if not transfer_ids:
            raise ValueError("shared KV consume set must not be empty")
        leases = []
        for transfer_id in transfer_ids:
            lease = self._pending_transfers.get(transfer_id)
            if lease is None:
                raise ValueError("unknown shared KV transfer")
            if self._states[lease.slot_id] not in {
                KVSlotState.READY,
                KVSlotState.CONSUMING,
            }:
                raise ValueError("shared KV transfer is not ready")
            leases.append(lease)
        self._consuming_transfers.update(transfer_ids)
        for lease in set(leases):
            self._states[lease.slot_id] = KVSlotState.CONSUMING
            self.handle.states[lease.slot_id] = _STATE_CODES[KVSlotState.CONSUMING]
            self._record_state(
                lease,
                KVSlotState.CONSUMING,
                writer="prefill_worker.ack",
            )

    def ack(self, transfer_id: str):
        lease = self._pending_transfers.get(transfer_id)
        if lease is None:
            raise ValueError("unknown transfer acknowledgement")
        if transfer_id not in self._consuming_transfers:
            raise ValueError("shared KV transfer is not consuming")
        del self._pending_transfers[transfer_id]
        self._consuming_transfers.remove(transfer_id)
        if lease not in self._pending_transfers.values():
            self._states[lease.slot_id] = KVSlotState.FREE
            self.handle.states[lease.slot_id] = _STATE_CODES[KVSlotState.FREE]
            self._leases.pop(lease.slot_id, None)
            self._record_state(
                lease,
                KVSlotState.FREE,
                writer="prefill_worker.ack",
            )

    def cancel(self, lease: KVSlotLease):
        active = self._leases.get(lease.slot_id)
        if active != lease:
            raise ValueError("shared KV lease is stale")
        for transfer_id, pending_lease in list(self._pending_transfers.items()):
            if pending_lease == lease:
                del self._pending_transfers[transfer_id]
                self._consuming_transfers.discard(transfer_id)
        self._states[lease.slot_id] = KVSlotState.FREE
        self.handle.states[lease.slot_id] = _STATE_CODES[KVSlotState.FREE]
        self._leases.pop(lease.slot_id, None)
        self._record_state(
            lease,
            KVSlotState.FREE,
            writer="prefill_worker.cancel",
        )

    def slot_state(self, slot_id: int) -> KVSlotState:
        if not isinstance(slot_id, int) or not 0 <= slot_id < len(self._states):
            raise ValueError("shared KV slot id is out of range")
        return self._states[slot_id]

    def stats(self) -> dict[str, int]:
        return {
            "slot_count": len(self._states),
            "free_slots": self._states.count(KVSlotState.FREE),
            "filling_slots": self._states.count(KVSlotState.FILLING),
            "ready_slots": self._states.count(KVSlotState.READY),
            "consuming_slots": self._states.count(KVSlotState.CONSUMING),
            "pending_transfers": len(self._pending_transfers),
        }

    def observability(self) -> dict:
        """Return owner-side slot transitions without changing slot ownership."""

        return {
            "environment": dict(self._environment),
            "events": [dict(event) for event in self._events],
            "state_durations_ms": {
                state.value: seconds * 1000
                for state, seconds in self._state_durations_seconds.items()
            },
            "acquire_exhaustions": self._acquire_exhaustions,
        }
