import unittest
import signal

from llmserve.pd.process import (
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
