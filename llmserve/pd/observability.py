"""Pure observability helpers shared by PD transport backends."""

from __future__ import annotations

from enum import Enum
import os
from pathlib import Path
import re
from typing import Callable, Iterable


class DecodeIdleReason(str, Enum):
    """Mutually exclusive reason why Decode made no progress in one interval."""

    NO_RUNNABLE_REQUEST = "no_runnable_request"
    WAITING_PREFILL_OUTPUT = "waiting_prefill_output"
    WAITING_KV_H2D = "waiting_kv_h2d"
    SCHEDULER_NOT_SCHEDULED = "scheduler_not_scheduled"


def classify_decode_idle_reason(
    *,
    active_decode_requests: int,
    pending_prefill_requests: int,
    prefill_inflight: bool,
    pending_kv_imports: int,
    scheduled_decode_requests: int | None = None,
) -> DecodeIdleReason | None:
    """Classify an idle Decode interval without guessing a root cause."""

    if active_decode_requests > 0:
        if scheduled_decode_requests == 0:
            return DecodeIdleReason.SCHEDULER_NOT_SCHEDULED
        return None
    if pending_kv_imports > 0:
        return DecodeIdleReason.WAITING_KV_H2D
    if prefill_inflight or pending_prefill_requests > 0:
        return DecodeIdleReason.WAITING_PREFILL_OUTPUT
    return DecodeIdleReason.NO_RUNNABLE_REQUEST


def _total_duration(intervals: Iterable[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def _validate_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    result = []
    for start, end in intervals:
        if end < start:
            raise ValueError("interval end must not precede its start")
        result.append((float(start), float(end)))
    return result


def summarize_copy_compute_overlap(
    *,
    copy_intervals: Iterable[tuple[float, float]],
    compute_intervals: Iterable[tuple[float, float]],
) -> dict[str, float]:
    """Summarize millisecond host intervals for transport/compute overlap."""

    copies = _validate_intervals(copy_intervals)
    computes = _validate_intervals(compute_intervals)
    copy_ms = _total_duration(copies)
    compute_ms = _total_duration(computes)
    overlap_ms = sum(
        max(0.0, min(copy_end, compute_end) - max(copy_start, compute_start))
        for copy_start, copy_end in copies
        for compute_start, compute_end in computes
    )
    all_intervals = copies + computes
    makespan_ms = (
        max(end for _, end in all_intervals) - min(start for start, _ in all_intervals)
        if all_intervals
        else 0.0
    )
    serial_ms = copy_ms + compute_ms
    return {
        "copy_ms": copy_ms,
        "compute_ms": compute_ms,
        "overlap_ms": overlap_ms,
        "copy_overlap_rate": overlap_ms / copy_ms if copy_ms else 0.0,
        "serial_ms": serial_ms,
        "makespan_ms": makespan_ms,
        "critical_path_reduction_ms": max(0.0, serial_ms - makespan_ms),
    }


def _proc_maps_range(proc_maps: str, address: int) -> tuple[int, int] | None:
    for line in proc_maps.splitlines():
        match = re.match(r"^([0-9a-fA-F]+)-([0-9a-fA-F]+)\s+", line)
        if match is None:
            continue
        start, end = (int(match.group(index), 16) for index in (1, 2))
        if start <= address < end:
            return start, end
    return None


def _numa_details(address_range: str, details: str) -> dict:
    policy = next(
        (token for token in details.split() if token in {"default", "interleave", "bind"}),
        None,
    )
    page_counts = {
        int(node): int(count)
        for node, count in re.findall(r"\bN(\d+)=(\d+)", details)
    }
    return {
        "address_range": address_range,
        "policy": policy,
        "page_counts": page_counts,
    }


def parse_numa_mapping(
    numa_maps: str,
    address: int,
    *,
    address_range: tuple[int, int] | None = None,
) -> dict | None:
    """Return NUMA placement for an address, optionally using ``/proc/*/maps``.

    Linux ``numa_maps`` normally lists only a mapping's start address. A caller
    that needs a reliable containment check must provide the matching virtual
    range from ``/proc/<pid>/maps``. The range form remains supported for tests
    and platform variants that expose it directly.
    """

    target_start = address_range[0] if address_range is not None else None
    for line in numa_maps.splitlines():
        range_match = re.match(
            r"^([0-9a-fA-F]+)-([0-9a-fA-F]+)\s+(.*)$", line
        )
        if range_match is not None:
            start, end = (int(range_match.group(index), 16) for index in (1, 2))
            if start <= address < end:
                return _numa_details(
                    f"{range_match.group(1)}-{range_match.group(2)}",
                    range_match.group(3),
                )
            continue
        start_match = re.match(r"^([0-9a-fA-F]+)\s+(.*)$", line)
        if start_match is None or target_start != int(start_match.group(1), 16):
            continue
        return _numa_details(
            f"{address_range[0]:x}-{address_range[1]:x}",
            start_match.group(2),
        )
    return None
    return None


def collect_process_numa_observability(
    *,
    pid: int | None = None,
    shared_memory_address: int | None = None,
    get_affinity: Callable[[int], set[int]] = os.sched_getaffinity,
    read_text: Callable[[str], str] | None = None,
) -> dict:
    """Collect Linux process placement without mutating CPU or memory policy."""

    pid = os.getpid() if pid is None else pid
    if read_text is None:
        read_text = lambda path: Path(path).read_text()
    try:
        affinity = sorted(get_affinity(pid))
    except (AttributeError, OSError):
        affinity = None
    mapping = None
    if shared_memory_address is not None:
        try:
            proc_maps = read_text(f"/proc/{pid}/maps")
            mapping = parse_numa_mapping(
                read_text(f"/proc/{pid}/numa_maps"),
                shared_memory_address,
                address_range=_proc_maps_range(proc_maps, shared_memory_address),
            )
        except OSError:
            mapping = None
    return {
        "pid": pid,
        "cpu_affinity": affinity,
        "shared_memory_numa": mapping,
    }
