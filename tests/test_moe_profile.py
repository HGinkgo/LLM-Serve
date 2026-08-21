from __future__ import annotations

import unittest

from benchmarks.moe_profile import (
    build_profile_engine_kwargs,
    summarize_profile_events,
)


class _FakeEvent:
    def __init__(
        self,
        key: str,
        *,
        count: int,
        self_cpu_us: float,
        cpu_us: float,
        self_device_us: float,
        device_us: float,
        device_type=None,
    ) -> None:
        self.key = key
        self.count = count
        self.self_cpu_time_total = self_cpu_us
        self.cpu_time_total = cpu_us
        self.self_device_time_total = self_device_us
        self.device_time_total = device_us
        if device_type is not None:
            self.device_type = device_type


class MoeProfileTest(unittest.TestCase):

    def test_summarize_profile_events_converts_and_sorts_device_time(self):
        events = [
            _FakeEvent(
                "llmserve.moe.dispatch",
                count=2,
                self_cpu_us=1200,
                cpu_us=2500,
                self_device_us=0,
                device_us=0,
            ),
            _FakeEvent(
                "aten::index_select",
                count=4,
                self_cpu_us=800,
                cpu_us=1500,
                self_device_us=40,
                device_us=90,
            ),
            _FakeEvent(
                "aten::index_select",
                count=4,
                self_cpu_us=200,
                cpu_us=300,
                self_device_us=5,
                device_us=7,
            ),
        ]

        summary = summarize_profile_events(events)

        self.assertEqual(
            [row["name"] for row in summary],
            ["aten::index_select", "llmserve.moe.dispatch"],
        )
        self.assertEqual(
            summary[0],
            {
                "name": "aten::index_select",
                "calls": 4,
                "self_cpu_ms": 1.0,
                "cpu_total_ms": 1.8,
                "self_cuda_ms": 0.045,
                "cuda_total_ms": 0.097,
            },
        )
        self.assertEqual(summary[1]["self_cuda_ms"], 0.0)

    def test_summarize_profile_events_does_not_double_count_device_duration(self):
        events = [
            _FakeEvent(
                "llmserve.gptq.marlin",
                count=4,
                self_cpu_us=100,
                cpu_us=300,
                self_device_us=0,
                device_us=90,
                device_type="DeviceType.CPU",
            ),
            _FakeEvent(
                "llmserve.gptq.marlin",
                count=4,
                self_cpu_us=0,
                cpu_us=0,
                self_device_us=40,
                device_us=90,
                device_type="DeviceType.CUDA",
            ),
        ]

        self.assertEqual(
            summarize_profile_events(events),
            [{
                "name": "llmserve.gptq.marlin",
                "calls": 4,
                "self_cpu_ms": 0.1,
                "cpu_total_ms": 0.3,
                "self_cuda_ms": 0.04,
                "cuda_total_ms": 0.09,
            }],
        )

    def test_summarize_profile_events_keeps_native_device_time_without_cuda_row(self):
        summary = summarize_profile_events([
            _FakeEvent(
                "aten::_weight_int4pack_mm",
                count=4,
                self_cpu_us=100,
                cpu_us=300,
                self_device_us=40,
                device_us=90,
                device_type="DeviceType.CPU",
            ),
        ])

        self.assertEqual(summary[0]["self_cuda_ms"], 0.04)
        self.assertEqual(summary[0]["cuda_total_ms"], 0.09)

    def test_build_profile_engine_kwargs_forwards_runtime_limits(self):
        kwargs = build_profile_engine_kwargs(
            {
                "max_model_len": 512,
                "max_num_batched_tokens": 128,
                "max_num_seqs": 4,
                "gpu_memory_utilization": 0.85,
                "seed": 20260820,
                "marlin_library": "/tmp/marlin.so",
            },
            backend="marlin",
        )

        self.assertEqual(
            kwargs,
            {
                "max_model_len": 512,
                "max_num_batched_tokens": 128,
                "max_num_seqs": 4,
                "gpu_memory_utilization": 0.85,
                "enforce_eager": True,
                "random_seed": 20260820,
                "gptq_backend": "marlin",
                "marlin_library": "/tmp/marlin.so",
                "enable_moe_gate_up_fusion": False,
            },
        )

    def test_build_profile_engine_kwargs_can_enable_gate_up_fusion(self):
        from benchmarks.moe_profile import build_profile_engine_kwargs

        kwargs = build_profile_engine_kwargs(
            {
                "max_model_len": 512,
                "max_num_batched_tokens": 128,
                "max_num_seqs": 4,
                "gpu_memory_utilization": 0.85,
                "enable_moe_gate_up_fusion": True,
            },
            backend="tinygemm",
        )

        self.assertTrue(kwargs["enable_moe_gate_up_fusion"])

    def test_build_profile_engine_kwargs_requires_marlin_library(self):
        with self.assertRaisesRegex(ValueError, "marlin_library"):
            build_profile_engine_kwargs(
                {
                    "max_model_len": 512,
                    "max_num_batched_tokens": 128,
                    "max_num_seqs": 4,
                    "gpu_memory_utilization": 0.85,
                },
                backend="marlin",
            )


if __name__ == "__main__":
    unittest.main()
