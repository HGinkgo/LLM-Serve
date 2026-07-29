import unittest

import torch

from llmserve.utils.stage_profiler import StageProfiler


class FakeClock:
    def __init__(self, values):
        self._values = iter(values)

    def __call__(self):
        return next(self._values)


class StageProfilerTests(unittest.TestCase):
    def test_cpu_stage_reports_host_and_device_time(self):
        profiler = StageProfiler(
            torch.device("cpu"),
            ("forward",),
            clock=FakeClock([1.0, 1.25]),
        )

        with profiler.stage("forward"):
            pass

        timing = profiler.finish()["forward"]
        self.assertAlmostEqual(timing.host_seconds, 0.25)
        self.assertAlmostEqual(timing.device_seconds, 0.25)

    def test_repeated_cpu_stage_accumulates_time(self):
        profiler = StageProfiler(
            torch.device("cpu"),
            ("forward",),
            clock=FakeClock([1.0, 1.1, 2.0, 2.2]),
        )

        with profiler.stage("forward"):
            pass
        with profiler.stage("forward"):
            pass

        timing = profiler.finish()["forward"]
        self.assertAlmostEqual(timing.host_seconds, 0.3)
        self.assertAlmostEqual(timing.device_seconds, 0.3)

    def test_unknown_stage_is_rejected(self):
        profiler = StageProfiler(torch.device("cpu"), ("forward",))

        with self.assertRaisesRegex(ValueError, "unknown profiler stage"):
            with profiler.stage("verify"):
                pass


if __name__ == "__main__":
    unittest.main()
