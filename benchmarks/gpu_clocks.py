"""Read-only GPU clock sampling for benchmark provenance."""

from __future__ import annotations

import subprocess
from collections import defaultdict
from threading import Event, Thread


def query_gpu_clocks():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,clocks.current.sm,clocks.current.memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    samples = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            samples.append({
                "gpu": int(fields[0]),
                "sm_mhz": int(fields[1]),
                "memory_mhz": int(fields[2]),
            })
        except ValueError:
            continue
    return samples


def summarize_clock_samples(samples):
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["gpu"]].append(sample)
    return {
        str(gpu): {
            "samples": len(group),
            "sm_mhz": {
                "min": min(sample["sm_mhz"] for sample in group),
                "max": max(sample["sm_mhz"] for sample in group),
            },
            "memory_mhz": {
                "min": min(sample["memory_mhz"] for sample in group),
                "max": max(sample["memory_mhz"] for sample in group),
            },
        }
        for gpu, group in sorted(grouped.items())
    }


class GPUClockSampler:
    """Sample clocks without changing clock policy."""

    def __init__(self, interval_seconds: float = 1.0, query=query_gpu_clocks):
        self.interval_seconds = interval_seconds
        self.query = query
        self.samples = []
        self._stop = Event()
        self._thread = None

    def _collect(self):
        while not self._stop.is_set():
            self.samples.extend(self.query())
            self._stop.wait(self.interval_seconds)

    def start(self):
        self.samples.extend(self.query())
        self._thread = Thread(target=self._collect, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 1.0)
        self.samples.extend(self.query())
        return summarize_clock_samples(self.samples)
