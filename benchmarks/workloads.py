from dataclasses import dataclass
from math import floor
from random import Random
from typing import Sequence


@dataclass(frozen=True)
class WorkloadClass:
    name: str
    weight: float
    input_len: int
    output_len: int

    def __post_init__(self):
        if not self.name:
            raise ValueError("workload class name cannot be empty")
        if self.weight <= 0:
            raise ValueError("workload class weight must be positive")
        if self.input_len <= 0:
            raise ValueError("input_len must be positive")
        if self.output_len <= 0:
            raise ValueError("output_len must be positive")


@dataclass(frozen=True)
class RequestSpec:
    request_id: int
    request_class: str
    input_len: int
    output_len: int
    prompt_token_ids: tuple[int, ...]


def _class_counts(
    classes: Sequence[WorkloadClass],
    num_requests: int,
) -> list[int]:
    total_weight = sum(workload_class.weight for workload_class in classes)
    exact = [
        num_requests * workload_class.weight / total_weight
        for workload_class in classes
    ]
    counts = [floor(value) for value in exact]
    remaining = num_requests - sum(counts)
    order = sorted(
        range(len(classes)),
        key=lambda index: (-(exact[index] - counts[index]), index),
    )
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def build_request_specs(
    classes: Sequence[WorkloadClass],
    num_requests: int,
    seed: int,
    ordering: str = "shuffled",
) -> list[RequestSpec]:
    if not classes:
        raise ValueError("classes cannot be empty")
    if num_requests < 0:
        raise ValueError("num_requests must be non-negative")

    rng = Random(seed)
    assignments = _ordered_assignments(
        classes,
        _class_counts(classes, num_requests),
        rng,
        ordering,
    )

    specs = []
    for request_id, workload_class in enumerate(assignments):
        prompt_token_ids = tuple(
            rng.randint(0, 10000) for _ in range(workload_class.input_len)
        )
        specs.append(
            RequestSpec(
                request_id=request_id,
                request_class=workload_class.name,
                input_len=workload_class.input_len,
                output_len=workload_class.output_len,
                prompt_token_ids=prompt_token_ids,
            )
        )
    return specs


def _ordered_assignments(
    classes: Sequence[WorkloadClass],
    counts: Sequence[int],
    rng: Random,
    ordering: str,
) -> list[WorkloadClass]:
    if ordering == "shuffled":
        assignments = []
        for workload_class, count in zip(classes, counts):
            assignments.extend([workload_class] * count)
        rng.shuffle(assignments)
        return assignments

    if ordering == "interleaved":
        assignments = []
        remaining = list(counts)
        while any(remaining):
            for index, workload_class in enumerate(classes):
                if remaining[index] <= 0:
                    continue
                assignments.append(workload_class)
                remaining[index] -= 1
        return assignments

    if ordering == "balanced_interleaved":
        if len(classes) != 2 or counts[0] != counts[1]:
            raise ValueError(
                "balanced_interleaved requires two equally weighted classes"
            )
        assignments = []
        for _ in range(counts[0] // 2):
            assignments.extend((classes[0], classes[1], classes[1], classes[0]))
        return assignments

    if ordering == "replica_balanced":
        if len(classes) != 2 or any(count % 2 for count in counts):
            raise ValueError(
                "replica_balanced requires two classes with even per-cycle counts"
            )
        replica_assignments = []
        for workload_class, count in zip(classes, counts):
            replica_assignments.extend([workload_class] * (count // 2))
        rng.shuffle(replica_assignments)
        assignments = []
        for workload_class in replica_assignments:
            assignments.extend((workload_class, workload_class))
        return assignments

    raise ValueError(
        "ordering must be shuffled, interleaved, balanced_interleaved, or "
        "replica_balanced"
    )


def iter_request_specs(
    classes: Sequence[WorkloadClass],
    seed: int,
    cycle_size: int = 100,
    ordering: str = "shuffled",
):
    if not classes:
        raise ValueError("classes cannot be empty")
    if cycle_size <= 0:
        raise ValueError("cycle_size must be positive")
    if ordering not in {
        "shuffled", "interleaved", "balanced_interleaved", "replica_balanced",
    }:
        raise ValueError("unsupported request ordering")

    rng = Random(seed)
    request_id = 0
    counts = _class_counts(classes, cycle_size)
    while True:
        assignments = _ordered_assignments(classes, counts, rng, ordering)
        for workload_class in assignments:
            prompt_token_ids = tuple(
                rng.randint(0, 10000)
                for _ in range(workload_class.input_len)
            )
            yield RequestSpec(
                request_id=request_id,
                request_class=workload_class.name,
                input_len=workload_class.input_len,
                output_len=workload_class.output_len,
                prompt_token_ids=prompt_token_ids,
            )
            request_id += 1
