import unittest
from collections import deque
from types import SimpleNamespace

from llmserve.pd.coordinator import PDConfig, PDCoordinator, PDWorkerError
from llmserve.pd.protocol import RequestEnvelope


class FakeQueue:

    def __init__(self, responses=()):
        self.responses = deque(responses)
        self.commands = []

    def put(self, command):
        self.commands.append(command)

    def get(self, timeout=None):
        return self.responses.popleft()


class TestPDConfig(unittest.TestCase):

    def test_coordinator_does_not_duplicate_serving_engine_loops(self):
        self.assertFalse(hasattr(PDCoordinator, "generate"))
        self.assertFalse(hasattr(PDCoordinator, "generate_many"))

    def test_builds_isolated_prefill_and_decode_engine_configs(self):
        config = PDConfig(
            model="/models/qwen3",
            prefill_gpu=0,
            decode_gpu=1,
            engine_kwargs={"max_num_seqs": 8},
        )

        prefill = config.engine_kwargs_for("prefill")
        decode = config.engine_kwargs_for("decode")

        self.assertEqual(prefill["tensor_parallel_size"], 1)
        self.assertEqual(decode["tensor_parallel_size"], 1)
        self.assertTrue(prefill["enforce_eager"])
        self.assertIsNone(prefill["speculative_model"])
        self.assertEqual(prefill["max_num_seqs"], 8)
        self.assertNotEqual(
            prefill["distributed_init_method"],
            decode["distributed_init_method"],
        )

    def test_worker_eager_modes_can_be_configured_independently(self):
        config = PDConfig(
            model="/models/qwen3",
            prefill_gpu=0,
            decode_gpu=1,
            prefill_enforce_eager=True,
            decode_enforce_eager=False,
        )

        self.assertTrue(config.engine_kwargs_for("prefill")["enforce_eager"])
        self.assertFalse(config.engine_kwargs_for("decode")["enforce_eager"])

    def test_rejects_same_gpu_for_both_workers(self):
        with self.assertRaises(ValueError):
            PDConfig(model="/models/qwen3", prefill_gpu=0, decode_gpu=0)

    def test_wait_worker_ready_consumes_readiness_response(self):
        config = PDConfig(model="/models/qwen3", prefill_gpu=0, decode_gpu=1)
        coordinator = PDCoordinator.__new__(PDCoordinator)
        coordinator.config = config
        coordinator._workers = {
            "prefill": {
                "responses": FakeQueue(
                    responses=[
                        {
                            "ok": True,
                            "result": {"ready": True, "role": "prefill"},
                        }
                    ]
                )
            }
        }

        coordinator._wait_worker_ready("prefill")

        self.assertTrue(coordinator._workers["prefill"]["ready"])

    def test_prefill_batch_serializes_envelopes_for_worker_rpc(self):
        config = PDConfig(model="/models/qwen3", prefill_gpu=0, decode_gpu=1)
        coordinator = PDCoordinator.__new__(PDCoordinator)
        coordinator.config = config
        coordinator._started = True
        queue = FakeQueue(responses=[{"ok": True, "result": ["handoff"]}])
        coordinator._workers = {
            "prefill": {"commands": queue, "responses": queue},
        }
        envelope = RequestEnvelope(7, (1, 2, 3), 4, 1.0, True)

        result = coordinator.prefill_batch([envelope])

        self.assertEqual(result, ["handoff"])
        self.assertEqual(queue.commands[0]["type"], "prefill_batch")
        self.assertEqual(queue.commands[0]["envelopes"], [envelope.to_payload()])

    def test_worker_rpc_surfaces_remote_error(self):
        config = PDConfig(model="/models/qwen3", prefill_gpu=0, decode_gpu=1)
        coordinator = PDCoordinator.__new__(PDCoordinator)
        coordinator.config = config
        coordinator._started = True
        queue = FakeQueue(
            responses=[
                {
                    "ok": False,
                    "error": {"type": "RuntimeError", "message": "prefill failed"},
                }
            ]
        )
        coordinator._workers = {
            "prefill": {"commands": queue, "responses": queue},
        }
        envelope = RequestEnvelope(7, (1, 2, 3), 4, 1.0, True)

        with self.assertRaisesRegex(PDWorkerError, "prefill failed"):
            coordinator.prefill_batch([envelope])

    def test_worker_health_reports_process_state(self):
        coordinator = PDCoordinator.__new__(PDCoordinator)
        coordinator._workers = {
            "prefill": {
                "process": SimpleNamespace(
                    pid=10,
                    exitcode=None,
                    is_alive=lambda: True,
                )
            },
            "decode": {
                "process": SimpleNamespace(
                    pid=11,
                    exitcode=0,
                    is_alive=lambda: False,
                )
            },
        }

        health = coordinator.worker_health()

        self.assertEqual(
            health["prefill"],
            {"pid": 10, "alive": True, "exitcode": None},
        )
        self.assertEqual(
            health["decode"],
            {"pid": 11, "alive": False, "exitcode": 0},
        )


if __name__ == "__main__":
    unittest.main()
