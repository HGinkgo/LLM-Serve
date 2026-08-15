import unittest
import signal

import torch

from llmserve.pd.process import (
    _create_prefill_slot_pools,
    _cleanup_worker_resources,
    _destroy_process_group,
    _handle_termination,
)


class FakeDistributed:

    def __init__(self, initialized):
        self.initialized = initialized
        self.destroy_calls = 0

    def is_initialized(self):
        return self.initialized

    def destroy_process_group(self):
        self.destroy_calls += 1
        self.initialized = False


class TestPDProcessCleanup(unittest.TestCase):

    def test_creates_an_isolated_shared_slot_pool_for_each_decode_worker(self):
        engine = type(
            "Engine",
            (), {
                "model_runner": type(
                    "Runner",
                    (), {"kv_cache": torch.empty(2, 1, 1, 4, 1, 2)},
                )(),
            },
        )()

        pools = _create_prefill_slot_pools(
            engine,
            {
                "slot_count": 2,
                "capacity_tokens": 8,
                "target_workers": ("decode-0", "decode-1"),
            },
            register_cuda=False,
        )

        self.assertEqual(set(pools), {"decode-0", "decode-1"})
        self.assertIsNot(pools["decode-0"], pools["decode-1"])
        self.assertNotEqual(
            pools["decode-0"].handle.backing.data_ptr(),
            pools["decode-1"].handle.backing.data_ptr(),
        )

    def test_sigterm_is_converted_to_system_exit_for_finally_cleanup(self):
        with self.assertRaises(SystemExit) as raised:
            _handle_termination(signal.SIGTERM, None)

        self.assertEqual(raised.exception.code, 128 + signal.SIGTERM)

    def test_destroys_initialized_process_group(self):
        distributed = FakeDistributed(initialized=True)

        _destroy_process_group(distributed)

        self.assertEqual(distributed.destroy_calls, 1)

    def test_ignores_uninitialized_process_group(self):
        distributed = FakeDistributed(initialized=False)

        _destroy_process_group(distributed)

        self.assertEqual(distributed.destroy_calls, 0)

    def test_cleanup_synchronizes_engine_before_closing_shared_transport(self):
        calls = []

        class Engine:

            def exit(self):
                calls.append("engine.exit")

        class Transport:

            def close(self):
                calls.append("transport.close")

        class Runtime:
            slot_reader = Transport()

        _cleanup_worker_resources(Engine(), Runtime())

        self.assertEqual(calls, ["engine.exit", "transport.close"])


if __name__ == "__main__":
    unittest.main()
