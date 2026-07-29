from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Iterable

import torch


@dataclass(frozen=True, slots=True)
class StageTiming:
    host_seconds: float
    device_seconds: float


class StageProfiler:
    def __init__(
        self,
        device: torch.device,
        stage_names: Iterable[str],
        *,
        clock: Callable[[], float] = perf_counter,
    ):
        self.device = torch.device(device)
        self._clock = clock
        self._host_seconds = {name: 0.0 for name in stage_names}
        self._device_seconds = {name: 0.0 for name in self._host_seconds}
        self._cuda_events = []

    @contextmanager
    def stage(self, name: str):
        if name not in self._host_seconds:
            raise ValueError(f"unknown profiler stage: {name}")

        host_start = self._clock()
        if self.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self._host_seconds[name] += self._clock() - host_start
                self._cuda_events.append((name, start, end))
            return

        try:
            yield
        finally:
            elapsed = self._clock() - host_start
            self._host_seconds[name] += elapsed
            self._device_seconds[name] += elapsed

    def finish(self) -> dict[str, StageTiming]:
        if self._cuda_events:
            torch.cuda.synchronize(self.device)
            for name, start, end in self._cuda_events:
                self._device_seconds[name] += start.elapsed_time(end) / 1000
            self._cuda_events.clear()

        return {
            name: StageTiming(
                host_seconds=self._host_seconds[name],
                device_seconds=self._device_seconds[name],
            )
            for name in self._host_seconds
        }
