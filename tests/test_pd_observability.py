import unittest


class TestPDObservability(unittest.TestCase):

    def test_classifies_decode_idle_reasons_from_runtime_state(self):
        from llmserve.pd.observability import (
            DecodeIdleReason,
            classify_decode_idle_reason,
        )

        self.assertEqual(
            classify_decode_idle_reason(
                active_decode_requests=0,
                pending_prefill_requests=0,
                prefill_inflight=False,
                pending_kv_imports=0,
            ),
            DecodeIdleReason.NO_RUNNABLE_REQUEST,
        )
        self.assertEqual(
            classify_decode_idle_reason(
                active_decode_requests=0,
                pending_prefill_requests=3,
                prefill_inflight=True,
                pending_kv_imports=0,
            ),
            DecodeIdleReason.WAITING_PREFILL_OUTPUT,
        )
        self.assertEqual(
            classify_decode_idle_reason(
                active_decode_requests=0,
                pending_prefill_requests=0,
                prefill_inflight=False,
                pending_kv_imports=2,
            ),
            DecodeIdleReason.WAITING_KV_H2D,
        )
        self.assertEqual(
            classify_decode_idle_reason(
                active_decode_requests=4,
                pending_prefill_requests=0,
                prefill_inflight=False,
                pending_kv_imports=0,
                scheduled_decode_requests=0,
            ),
            DecodeIdleReason.SCHEDULER_NOT_SCHEDULED,
        )

    def test_overlap_summary_reports_rate_and_serial_critical_path(self):
        from llmserve.pd.observability import summarize_copy_compute_overlap

        summary = summarize_copy_compute_overlap(
            copy_intervals=[(0.0, 4.0), (8.0, 10.0)],
            compute_intervals=[(2.0, 6.0), (9.0, 13.0)],
        )

        self.assertEqual(summary["copy_ms"], 6.0)
        self.assertEqual(summary["compute_ms"], 8.0)
        self.assertEqual(summary["overlap_ms"], 3.0)
        self.assertEqual(summary["copy_overlap_rate"], 0.5)
        self.assertEqual(summary["serial_ms"], 14.0)
        self.assertEqual(summary["makespan_ms"], 13.0)
        self.assertEqual(summary["critical_path_reduction_ms"], 1.0)

    def test_parses_shared_mapping_numa_page_counts(self):
        from llmserve.pd.observability import parse_numa_mapping

        mapping = parse_numa_mapping(
            "1000-2000 default file=/dev/shm/torch_anon dirty=8 N0=3 N1=5\n"
            "3000-4000 default anon=2 N0=2\n",
            0x1800,
        )

        self.assertEqual(mapping["address_range"], "1000-2000")
        self.assertEqual(mapping["page_counts"], {0: 3, 1: 5})
        self.assertEqual(mapping["policy"], "default")

    def test_collects_affinity_and_shared_page_placement(self):
        from llmserve.pd.observability import collect_process_numa_observability

        snapshot = collect_process_numa_observability(
            pid=123,
            shared_memory_address=0x1800,
            get_affinity=lambda pid: {0, 1, 5},
            read_text=lambda path: (
                "1000-2000 default file=/dev/shm/slot dirty=8 N0=3 N1=5\n"
            ),
        )

        self.assertEqual(snapshot["pid"], 123)
        self.assertEqual(snapshot["cpu_affinity"], [0, 1, 5])
        self.assertEqual(snapshot["shared_memory_numa"]["page_counts"], {0: 3, 1: 5})

    def test_collect_uses_proc_maps_for_real_numa_maps_format(self):
        from llmserve.pd.observability import collect_process_numa_observability

        def read_text(path):
            if path.endswith("/maps"):
                return "1000-2000 rw-s 00000000 00:01 42 /dev/shm/slot\n"
            return "1000 default file=/dev/shm/slot dirty=8 N0=3 N1=5\n"

        snapshot = collect_process_numa_observability(
            pid=123,
            shared_memory_address=0x1800,
            get_affinity=lambda pid: {0, 1},
            read_text=read_text,
        )

        self.assertEqual(
            snapshot["shared_memory_numa"]["address_range"],
            "1000-2000",
        )
        self.assertEqual(
            snapshot["shared_memory_numa"]["page_counts"],
            {0: 3, 1: 5},
        )


if __name__ == "__main__":
    unittest.main()
