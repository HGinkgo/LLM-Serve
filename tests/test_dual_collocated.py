import unittest
from copy import deepcopy


class FakeCoordinator:
    worker_ids = ("replica-0", "replica-1")

    def __init__(self):
        self.add_calls = []
        self.record_calls = []
        self.step_calls = []
        self.closed = False
        self.responses = {
            "replica-0": {
                "outputs": [],
                "num_tokens": 0,
                "last_step_events": {},
            },
            "replica-1": {
                "outputs": [],
                "num_tokens": 0,
                "last_step_events": {},
            },
        }
        self.metrics = {
            worker_id: {"summary": {}, "requests": []}
            for worker_id in self.worker_ids
        }

    def add_request(self, worker_id, prompt_token_ids, sampling_params, submitted_at):
        local_seq_id = 100 * (self.worker_ids.index(worker_id) + 1) + len(
            [item for item in self.add_calls if item[0] == worker_id]
        )
        self.add_calls.append(
            (worker_id, tuple(prompt_token_ids), sampling_params, submitted_at, local_seq_id)
        )
        return local_seq_id

    def record_benchmark_submit(self, worker_id, local_seq_id, submitted_at):
        self.record_calls.append((worker_id, local_seq_id, submitted_at))

    def step_all(self, worker_ids):
        self.step_calls.append(tuple(worker_ids))
        return {
            worker_id: deepcopy(self.responses[worker_id])
            for worker_id in worker_ids
        }

    def metrics_all(self):
        return deepcopy(self.metrics)

    def worker_health(self):
        return {
            worker_id: {"alive": True, "exitcode": None}
            for worker_id in self.worker_ids
        }

    def reset_metrics_all(self):
        return {worker_id: {"reset": True} for worker_id in self.worker_ids}

    def close(self):
        self.closed = True


class DualCollocatedServingEngineTests(unittest.TestCase):
    def test_config_uses_two_unique_replicas_and_endpoints(self):
        from benchmarks.dual_collocated import DualCollocatedConfig

        config = DualCollocatedConfig(
            model="Qwen3-8B",
            gpu_ids=(0, 1),
            init_methods=(
                "tcp://127.0.0.1:24711",
                "tcp://127.0.0.1:24712",
            ),
            engine_kwargs={"enforce_eager": False},
        )

        self.assertEqual(config.worker_ids, ("replica-0", "replica-1"))
        self.assertEqual(config.engine_kwargs_for("replica-0")["distributed_init_method"], "tcp://127.0.0.1:24711")
        self.assertEqual(config.engine_kwargs_for("replica-1")["distributed_init_method"], "tcp://127.0.0.1:24712")
        self.assertFalse(config.engine_kwargs_for("replica-0")["enforce_eager"])

    def _engine(self):
        from benchmarks.dual_collocated import DualCollocatedServingEngine

        coordinator = FakeCoordinator()
        return DualCollocatedServingEngine(coordinator), coordinator

    def test_round_robin_routes_requests_to_both_replicas(self):
        engine, coordinator = self._engine()

        request_ids = [
            engine.add_benchmark_request([index], object(), 10.0 + index)
            for index in range(8)
        ]

        self.assertEqual(request_ids, list(range(8)))
        self.assertEqual(
            [call[0] for call in coordinator.add_calls],
            ["replica-0", "replica-1"] * 4,
        )
        self.assertEqual(
            engine.request_assignments(),
            {
                0: "replica-0",
                1: "replica-1",
                2: "replica-0",
                3: "replica-1",
                4: "replica-0",
                5: "replica-1",
                6: "replica-0",
                7: "replica-1",
            },
        )

    def test_step_aggregates_worker_events_and_maps_finished_outputs(self):
        engine, coordinator = self._engine()
        first = engine.add_benchmark_request([1], object(), 1.0)
        second = engine.add_benchmark_request([2], object(), 1.0)
        first_local = coordinator.add_calls[0][-1]
        second_local = coordinator.add_calls[1][-1]
        coordinator.responses["replica-0"] = {
            "outputs": [(first_local, [11, 12])],
            "num_tokens": 2,
            "last_step_events": {
                "step_start": 2.0,
                "step_end": 2.2,
                "scheduled_seq_ids": [first_local],
                "prefill_seq_ids": [first_local],
                "finished_seq_ids": [first_local],
                "waiting_queue_size": 1,
                "running_queue_size": 3,
                "prefill_token_count": 1,
                "decode_token_count": 0,
            },
        }
        coordinator.responses["replica-1"] = {
            "outputs": [(second_local, [21])],
            "num_tokens": 1,
            "last_step_events": {
                "step_start": 2.1,
                "step_end": 2.3,
                "scheduled_seq_ids": [second_local],
                "decode_seq_ids": [second_local],
                "finished_seq_ids": [second_local],
                "waiting_queue_size": 2,
                "running_queue_size": 4,
                "prefill_token_count": 0,
                "decode_token_count": 1,
            },
        }

        outputs, num_tokens = engine.step()

        self.assertEqual(coordinator.step_calls, [("replica-0", "replica-1")])
        self.assertEqual(outputs, [(first, [11, 12]), (second, [21])])
        self.assertEqual(num_tokens, 3)
        self.assertTrue(engine.is_finished())
        self.assertEqual(engine.last_step_events["scheduled_seq_ids"], [first, second])
        self.assertEqual(engine.last_step_events["finished_seq_ids"], [first, second])
        self.assertEqual(engine.last_step_events["waiting_queue_size"], 3)
        self.assertEqual(engine.last_step_events["running_queue_size"], 7)
        self.assertEqual(engine.last_step_events["step_start"], 2.0)
        self.assertEqual(engine.last_step_events["step_end"], 2.3)
        self.assertEqual(
            engine.last_step_events["replica_queue_state"],
            {
                "replica-0": {"running_queue_size": 3, "decode_request_count": 0},
                "replica-1": {"running_queue_size": 4, "decode_request_count": 1},
            },
        )

    def test_step_records_zero_queue_for_an_idle_replica(self):
        engine, coordinator = self._engine()
        request_id = engine.add_benchmark_request([1], object(), 1.0)
        local_seq_id = coordinator.add_calls[0][-1]
        coordinator.responses["replica-0"] = {
            "outputs": [(local_seq_id, [11])],
            "num_tokens": 1,
            "last_step_events": {
                "scheduled_seq_ids": [local_seq_id],
                "finished_seq_ids": [local_seq_id],
                "running_queue_size": 1,
            },
        }

        outputs, _ = engine.step()

        self.assertEqual(outputs, [(request_id, [11])])
        self.assertEqual(
            engine.last_step_events["replica_queue_state"]["replica-1"],
            {"running_queue_size": 0, "decode_request_count": 0},
        )

    def test_metrics_rewrite_worker_local_ids_and_keep_replica_breakdown(self):
        engine, coordinator = self._engine()
        request_id = engine.add_benchmark_request([1], object(), 1.0)
        local_seq_id = coordinator.add_calls[0][-1]
        coordinator.metrics["replica-0"] = {
            "summary": {"cuda_graph": {"enabled": True}},
            "requests": [{"seq_id": local_seq_id, "success": True}],
        }

        metrics = engine.get_metrics()

        self.assertEqual(metrics["requests"][0]["seq_id"], request_id)
        self.assertEqual(metrics["requests"][0]["replica_id"], "replica-0")
        self.assertEqual(
            metrics["summary"]["replicas"]["replica-0"]["request_count"], 1
        )
        self.assertEqual(
            metrics["summary"]["routing"]["policy"],
            "round_robin",
        )

    def test_exit_closes_both_workers(self):
        engine, coordinator = self._engine()

        engine.exit()

        self.assertTrue(coordinator.closed)


if __name__ == "__main__":
    unittest.main()
