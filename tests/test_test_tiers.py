import unittest


class TestTierTests(unittest.TestCase):
    def test_core_tier_keeps_runtime_paths_and_excludes_experiment_tools(self):
        from tests.tier_runner import modules_for_tier

        core = set(modules_for_tier("core"))

        self.assertIn("tests.test_scheduler_speculative", core)
        self.assertIn("tests.test_pd_serving", core)
        self.assertIn("tests.test_service_api", core)
        self.assertIn("tests.test_moe_gptq", core)
        self.assertNotIn("tests.test_benchmark_serve", core)

    def test_extended_tier_is_a_non_overlapping_superset(self):
        from tests.tier_runner import modules_for_tier

        core = set(modules_for_tier("core"))
        extended = set(modules_for_tier("extended"))

        self.assertTrue(core < extended)
        self.assertEqual(len(extended), len(modules_for_tier("extended")))
        self.assertIn("tests.test_benchmark_serve", extended)
        self.assertIn("tests.test_moe_reference_parity", extended)
        self.assertIn("tests.test_moe_backend_benchmark", extended)
        self.assertIn("tests.test_moe_gptq_gemm_benchmark", extended)
        self.assertIn("tests.test_moe_profile", extended)
        self.assertIn("tests.test_moe_gate_up_summary", extended)
        self.assertIn("tests.test_service_overload_benchmark", extended)
        self.assertIn("tests.test_service_startup", extended)
        self.assertIn("tests.test_moe_gptq", extended)

    def test_gpu_tier_is_limited_to_real_model_or_cuda_coverage(self):
        from tests.tier_runner import modules_for_tier

        gpu = set(modules_for_tier("gpu"))

        self.assertIn("tests.test_model_runner_speculative", gpu)
        self.assertIn("tests.test_target_verify_graph", gpu)
        self.assertIn("tests.test_moe_gptq", gpu)
        self.assertNotIn("tests.test_benchmark_serve", gpu)


if __name__ == "__main__":
    unittest.main()
