import unittest
from unittest.mock import patch


class ServiceFactoryTests(unittest.TestCase):
    def test_service_launch_config_validates_admission_limits(self):
        from llmserve.service.factory import ServiceLaunchConfig

        with self.assertRaisesRegex(ValueError, "max_inflight_requests"):
            ServiceLaunchConfig(
                model="/models/Qwen3-8B",
                max_inflight_requests=0,
            )
        with self.assertRaisesRegex(ValueError, "request_timeout_seconds"):
            ServiceLaunchConfig(
                model="/models/Qwen3-8B",
                request_timeout_seconds=0.0,
            )

    def test_service_runtime_resolves_admission_capacity_per_deployment(self):
        from llmserve.service.factory import (
            ServiceLaunchConfig,
            build_service_runtime,
        )

        tokenizer = object()
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ):
            collocated = build_service_runtime(ServiceLaunchConfig(
                model="/models/Qwen3-8B",
                max_num_seqs=32,
            ))
            pd_shared = build_service_runtime(ServiceLaunchConfig(
                model="/models/Qwen3-8B",
                mode="pd-shared",
                max_num_seqs=32,
                decode_gpus=(1, 2),
                request_timeout_seconds=12.5,
            ))

        self.assertEqual(collocated._max_inflight_requests, 32)
        self.assertEqual(pd_shared._max_inflight_requests, 64)
        self.assertEqual(pd_shared._request_timeout_seconds, 12.5)

    def test_pd_shared_rejects_invalid_worker_and_slot_topology_before_startup(self):
        from llmserve.service.factory import ServiceLaunchConfig

        with self.assertRaisesRegex(ValueError, "at least two"):
            ServiceLaunchConfig(
                model="/models/Qwen3-8B",
                mode="pd-shared",
                kv_slot_count=1,
            )
        with self.assertRaisesRegex(ValueError, "different GPUs"):
            ServiceLaunchConfig(
                model="/models/Qwen3-8B",
                mode="pd-shared",
                prefill_gpu=0,
                decode_gpus=(0,),
            )

    def test_collocated_factory_forwards_serving_and_speculative_options(self):
        from llmserve.service.factory import ServiceLaunchConfig, build_engine_factory

        config = ServiceLaunchConfig(
            model="/models/Qwen3-8B",
            mode="collocated",
            max_model_len=1024,
            max_num_batched_tokens=512,
            max_num_seqs=32,
            gpu_memory_utilization=0.8,
            enable_chunked_prefill=True,
            enable_kv_capacity_admission=True,
            enforce_eager=False,
            speculative_model="/models/Qwen3-8B-EAGLE3",
            speculative_gamma=3,
        )
        expected_engine = object()
        with patch("llmserve.service.factory.LLM", return_value=expected_engine) as llm:
            engine = build_engine_factory(config)()

        self.assertIs(engine, expected_engine)
        llm.assert_called_once_with(
            "/models/Qwen3-8B",
            max_model_len=1024,
            max_num_batched_tokens=512,
            max_num_seqs=32,
            gpu_memory_utilization=0.8,
            enable_chunked_prefill=True,
            enable_kv_capacity_admission=True,
            enforce_eager=False,
            speculative_model="/models/Qwen3-8B-EAGLE3",
            speculative_gamma=3,
        )

    def test_pd_shared_factory_uses_distinct_workers_and_shared_slots(self):
        from llmserve.service.factory import ServiceLaunchConfig, build_engine_factory

        config = ServiceLaunchConfig(
            model="/models/Qwen3-8B",
            mode="pd-shared",
            max_model_len=2048,
            max_num_batched_tokens=1024,
            max_num_seqs=64,
            prefill_gpu=2,
            decode_gpus=(3,),
            prefill_batch_size=4,
            kv_slot_count=2,
            kv_slot_capacity_tokens=8192,
            startup_timeout_seconds=300.0,
        )
        coordinator = object()
        expected_engine = object()
        with (
            patch("llmserve.service.factory.PDCoordinator", return_value=coordinator) as coordinator_class,
            patch("llmserve.service.factory.PDServingEngine", return_value=expected_engine) as engine_class,
        ):
            engine = build_engine_factory(config)()

        self.assertIs(engine, expected_engine)
        pd_config = coordinator_class.call_args.args[0]
        self.assertEqual(pd_config.model, "/models/Qwen3-8B")
        self.assertEqual(pd_config.prefill_gpu, 2)
        self.assertEqual(pd_config.decode_gpus, (3,))
        self.assertEqual(pd_config.kv_slot_count, 2)
        self.assertEqual(pd_config.kv_slot_capacity_tokens, 8192)
        self.assertEqual(pd_config.startup_timeout_seconds, 300.0)
        self.assertEqual(pd_config.engine_kwargs["max_model_len"], 2048)
        self.assertEqual(pd_config.engine_kwargs["max_num_batched_tokens"], 1024)
        self.assertIsNone(pd_config.engine_kwargs["speculative_model"])
        engine_class.assert_called_once_with(
            coordinator,
            prefill_batch_size=4,
            enable_transport_overlap=True,
        )


if __name__ == "__main__":
    unittest.main()
