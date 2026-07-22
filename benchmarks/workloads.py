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
) -> list[RequestSpec]:
    if not classes:
        raise ValueError("classes cannot be empty")
    if num_requests < 0:
        raise ValueError("num_requests must be non-negative")

    rng = Random(seed)
    assignments = []
    for workload_class, count in zip(classes, _class_counts(classes, num_requests)):
        assignments.extend([workload_class] * count)
    rng.shuffle(assignments)

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
