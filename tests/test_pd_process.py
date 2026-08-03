import unittest

from llmserve.pd.process import _destroy_process_group


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

    def test_destroys_initialized_process_group(self):
        distributed = FakeDistributed(initialized=True)

        _destroy_process_group(distributed)

        self.assertEqual(distributed.destroy_calls, 1)

    def test_ignores_uninitialized_process_group(self):
        distributed = FakeDistributed(initialized=False)

        _destroy_process_group(distributed)

        self.assertEqual(distributed.destroy_calls, 0)


if __name__ == "__main__":
    unittest.main()
