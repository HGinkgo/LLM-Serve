import unittest
from collections import deque
from types import SimpleNamespace

from llmserve.pd.coordinator import PDConfig, PDCoordinator, PDWorkerError
from llmserve.pd.protocol import RequestEnvelope


class FakeQueue:

    def __init__(self, responses=()):
        self.responses = deque(responses)
        self.commands = []
        self.closed = False

    def put(self, command):
        self.commands.append(command)

    def get(self, timeout=None):
        return self.responses.popleft()

    def close(self):
        self.closed = True


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

    def test_config_assigns_unique_roles_and_endpoints_to_decode_pool(self):
        config = PDConfig(
            model="/models/qwen3",
            prefill_gpu=0,
            decode_gpu=1,
            decode_gpus=(1, 2),
            prefill_init_method="tcp://127.0.0.1:24431",
            decode_init_methods=(
                "tcp://127.0.0.1:24432",
                "tcp://127.0.0.1:24433",
            ),
            decode_enforce_eager=False,
        )

        self.assertEqual(config.decode_worker_ids, ("decode-0", "decode-1"))
        self.assertEqual(
            config.worker_specs(),
            (
                ("prefill", 0, "tcp://127.0.0.1:24431"),
                ("decode-0", 1, "tcp://127.0.0.1:24432"),
                ("decode-1", 2, "tcp://127.0.0.1:24433"),
            ),
        )
        self.assertFalse(config.engine_kwargs_for("decode-1")["enforce_eager"])
        self.assertEqual(
            config.engine_kwargs_for("decode-1")["distributed_init_method"],
            "tcp://127.0.0.1:24433",
        )
        self.assertEqual(
            config.prefill_transport_config(),
            {
                "slot_count": 2,
                "capacity_tokens": 1024,
                "target_workers": ("decode-0", "decode-1"),
            },
        )

    def test_config_rejects_duplicate_decode_pool_gpu_or_endpoint(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            PDConfig(
                model="/models/qwen3",
                prefill_gpu=0,
                decode_gpu=1,
                decode_gpus=(1, 1),
            )

        with self.assertRaisesRegex(ValueError, "different"):
            PDConfig(
                model="/models/qwen3",
                prefill_gpu=0,
                decode_gpu=1,
                decode_gpus=(1, 2),
                decode_init_methods=(
                    "tcp://127.0.0.1:24432",
                    "tcp://127.0.0.1:24432",
                ),
            )

    def test_shared_slot_config_is_transport_only_and_validated(self):
        config = PDConfig(
            model="/models/qwen3",
            prefill_gpu=0,
            decode_gpu=1,
            kv_slot_count=2,
            kv_slot_capacity_tokens=1024,
        )

        self.assertEqual(
            config.transport_config(),
            {"slot_count": 2, "capacity_tokens": 1024},
        )
        self.assertNotIn("kv_slot_count", config.engine_kwargs_for("prefill"))

        with self.assertRaisesRegex(ValueError, "slot count"):
            PDConfig(
                model="/models/qwen3",
                prefill_gpu=0,
                decode_gpu=1,
                kv_slot_count=1,
            )

        disabled = PDConfig(
            model="/models/qwen3",
            prefill_gpu=0,
            decode_gpu=1,
            kv_slot_count=0,
        )
        self.assertEqual(disabled.transport_config()["slot_count"], 0)

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

    def test_worker_health_retains_worker_environment_snapshot(self):
        coordinator = PDCoordinator.__new__(PDCoordinator)
        coordinator._workers = {
            "prefill": {
                "ready_result": {
                    "environment": {"cpu_affinity": [0, 1]},
                },
            },
            "decode": {
                "ready_result": {
                    "environment": {"cpu_affinity": [2, 3]},
                },
            },
        }

        health = coordinator.worker_health()

        self.assertEqual(health["prefill"]["environment"]["cpu_affinity"], [0, 1])
        self.assertEqual(health["decode"]["environment"]["cpu_affinity"], [2, 3])

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

        result = coordinator.prefill_batch(
            [envelope],
            release_transfer_ids=["transfer-previous"],
        )

        self.assertEqual(result, ["handoff"])
        self.assertEqual(queue.commands[0]["type"], "prefill_batch")
        self.assertEqual(queue.commands[0]["envelopes"], [envelope.to_payload()])
        self.assertEqual(
            queue.commands[0]["release_transfer_ids"],
            ["transfer-previous"],
        )
        timing = coordinator.last_rpc_timing("prefill")
        self.assertGreaterEqual(timing["roundtrip_ms"], 0.0)
        self.assertIn("parent_queue_put_ms", timing)

    def test_admit_batch_tracks_descriptor_only_transfer_ids(self):
        config = PDConfig(model="/models/qwen3", prefill_gpu=0, decode_gpu=1)
        coordinator = PDCoordinator.__new__(PDCoordinator)
        coordinator.config = config
        coordinator._started = True
        coordinator._last_rpc_timing = {}
        queue = FakeQueue(
            responses=[
                {
                    "ok": True,
                    "result": [{"seq_id": 9, "transfer_id": "transfer-7"}],
                }
            ]
        )
        coordinator._workers = {
            "decode": {"commands": queue, "responses": queue},
        }

        result = coordinator.admit_batch(["descriptor-only-handoff"])

        self.assertEqual(result[0]["transfer_id"], "transfer-7")
        self.assertEqual(queue.commands[0]["type"], "admit_batch")

    def test_attaches_each_decode_worker_to_its_own_shared_slot_pool(self):
        config = PDConfig(
            model="/models/qwen3",
            prefill_gpu=0,
            decode_gpu=1,
            decode_gpus=(1, 2),
        )
        coordinator = PDCoordinator.__new__(PDCoordinator)
        coordinator.config = config
        coordinator._started = True
        coordinator._last_rpc_timing = {}
        coordinator._failed_roles = set()
        first = FakeQueue(responses=[{"ok": True, "result": {"attached": True}}])
        second = FakeQueue(responses=[{"ok": True, "result": {"attached": True}}])
        coordinator._workers = {
            "decode-0": {"commands": first, "responses": first},
            "decode-1": {"commands": second, "responses": second},
        }

        coordinator._attach_decode_slot_pools(
            {"decode-0": "handle-0", "decode-1": "handle-1"}
        )

        self.assertEqual(first.commands[0]["handle"], "handle-0")
        self.assertEqual(second.commands[0]["handle"], "handle-1")

    def test_decode_pool_dispatches_every_step_before_waiting_for_replies(self):
        config = PDConfig(
            model="/models/qwen3",
            prefill_gpu=0,
            decode_gpu=1,
            decode_gpus=(1, 2),
        )
        coordinator = PDCoordinator.__new__(PDCoordinator)
        coordinator.config = config
        coordinator._started = True
        coordinator._last_rpc_timing = {}
        coordinator._failed_roles = set()
        first = FakeQueue(
            responses=[{"ok": True, "result": {"outputs": [], "num_tokens": 0}}]
        )
        second = FakeQueue(
            responses=[{"ok": True, "result": {"outputs": [], "num_tokens": 0}}]
        )
        coordinator._workers = {
            "decode-0": {"commands": first, "responses": first},
            "decode-1": {"commands": second, "responses": second},
        }

        results = coordinator.decode_step_all()

        self.assertEqual(set(results), {"decode-0", "decode-1"})
        self.assertEqual(first.commands[0]["type"], "step")
        self.assertEqual(second.commands[0]["type"], "step")

    def test_decode_pool_combines_targeted_handoffs_with_other_decode_steps(self):
        config = PDConfig(
            model="/models/qwen3",
            prefill_gpu=0,
            decode_gpu=1,
            decode_gpus=(1, 2),
        )
        coordinator = PDCoordinator.__new__(PDCoordinator)
        coordinator.config = config
        coordinator._started = True
        coordinator._last_rpc_timing = {}
        coordinator._failed_roles = set()
        first = FakeQueue(
            responses=[{"ok": True, "result": {"admissions": [], "outputs": []}}]
        )
        second = FakeQueue(
            responses=[{"ok": True, "result": {"outputs": []}}]
        )
        coordinator._workers = {
            "decode-0": {"commands": first, "responses": first},
            "decode-1": {"commands": second, "responses": second},
        }

        coordinator.decode_step_with_handoffs_all(
            {"decode-0": ["handoff-0"]},
            active_worker_ids=("decode-0", "decode-1"),
        )

        self.assertEqual(first.commands[0]["type"], "step_with_handoffs")
        self.assertEqual(first.commands[0]["handoffs"], ["handoff-0"])
        self.assertEqual(second.commands[0]["type"], "step")

    def test_decode_step_with_handoffs_uses_combined_decode_worker_rpc(self):
        config = PDConfig(model="/models/qwen3", prefill_gpu=0, decode_gpu=1)
        coordinator = PDCoordinator.__new__(PDCoordinator)
        coordinator.config = config
        coordinator._started = True
        coordinator._last_rpc_timing = {}
        queue = FakeQueue(
            responses=[
                {
                    "ok": True,
                    "result": {"admissions": [], "outputs": [], "num_tokens": 0},
                }
            ]
        )
        coordinator._workers = {
            "decode": {"commands": queue, "responses": queue},
        }

        result = coordinator.decode_step_with_handoffs(["shared-handoff"])

        self.assertEqual(result["num_tokens"], 0)
        self.assertEqual(queue.commands[0]["type"], "step_with_handoffs")
        self.assertEqual(queue.commands[0]["handoffs"], ["shared-handoff"])
        self.assertIn("_rpc_parent_sent_at", queue.commands[0])

    def test_abort_decode_request_sends_worker_command(self):
        config = PDConfig(model="/models/qwen3", prefill_gpu=0, decode_gpu=1)
        coordinator = PDCoordinator.__new__(PDCoordinator)
        coordinator.config = config
        coordinator._started = True
        coordinator._last_rpc_timing = {}
        queue = FakeQueue(responses=[{"ok": True, "result": True}])
        coordinator._workers = {
            "decode": {"commands": queue, "responses": queue},
        }

        self.assertTrue(coordinator.abort_decode_request(9))
        self.assertEqual(queue.commands[0]["type"], "abort_request")
        self.assertEqual(queue.commands[0]["seq_id"], 9)

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

    def test_rpc_fails_before_enqueue_when_worker_is_dead(self):
        config = PDConfig(model="/models/qwen3", prefill_gpu=0, decode_gpu=1)
        coordinator = PDCoordinator.__new__(PDCoordinator)
        coordinator.config = config
        coordinator._started = True
        coordinator._last_rpc_timing = {}
        queue = FakeQueue()
        coordinator._workers = {
            "decode": {
                "process": SimpleNamespace(is_alive=lambda: False, exitcode=9),
                "commands": queue,
                "responses": queue,
            },
        }

        with self.assertRaisesRegex(PDWorkerError, "not alive"):
            coordinator.decode_step()

        self.assertEqual(queue.commands, [])

    def test_close_skips_shutdown_rpc_for_failed_worker_and_is_idempotent(self):
        class FakeProcess:
            pid = 10
            exitcode = 1

            def __init__(self):
                self.join_calls = 0
                self.terminate_calls = 0

            def is_alive(self):
                return False

            def join(self, timeout=None):
                self.join_calls += 1

            def terminate(self):
                self.terminate_calls += 1

        config = PDConfig(model="/models/qwen3", prefill_gpu=0, decode_gpu=1)
        coordinator = PDCoordinator.__new__(PDCoordinator)
        coordinator.config = config
        coordinator._started = True
        coordinator._last_rpc_timing = {}
        coordinator._failed_roles = {"decode"}
        process = FakeProcess()
        commands = FakeQueue()
        responses = FakeQueue()
        coordinator._workers = {
            "decode": {
                "process": process,
                "commands": commands,
                "responses": responses,
            },
        }
        coordinator._transport_handle = object()

        coordinator.close()
        coordinator.close()

        self.assertEqual(commands.commands, [])
        self.assertEqual(process.join_calls, 1)
        self.assertTrue(commands.closed)
        self.assertTrue(responses.closed)
        self.assertFalse(coordinator._started)

    def test_rpc_wraps_broken_response_channel_as_worker_error(self):
        class BrokenResponseQueue(FakeQueue):
            def get(self, timeout=None):
                raise EOFError("response pipe closed")

        config = PDConfig(model="/models/qwen3", prefill_gpu=0, decode_gpu=1)
        coordinator = PDCoordinator.__new__(PDCoordinator)
        coordinator.config = config
        coordinator._started = True
        coordinator._last_rpc_timing = {}
        coordinator._failed_roles = set()
        commands = FakeQueue()
        coordinator._workers = {
            "decode": {
                "commands": commands,
                "responses": BrokenResponseQueue(),
            },
        }

        with self.assertRaisesRegex(PDWorkerError, "response channel failed"):
            coordinator.decode_step()

        self.assertEqual(coordinator._failed_roles, {"decode"})


if __name__ == "__main__":
    unittest.main()
