"""Named CPU and GPU test tiers for local development and CI."""

from __future__ import annotations

import argparse
import unittest


CORE_MODULES = (
    "tests.test_block_manager_speculative",
    "tests.test_check_speculative_correctness",
    "tests.test_cli",
    "tests.test_config_speculative",
    "tests.test_eagle3_cycle",
    "tests.test_eagle3_speculator",
    "tests.test_eagle3_target_bridge",
    "tests.test_eagle3_verify",
    "tests.test_llm_engine_speculative",
    "tests.test_linear_loading",
    "tests.test_model_runner_speculative",
    "tests.test_moe_gptq",
    "tests.test_pd_coordinator",
    "tests.test_pd_engine_bridge",
    "tests.test_pd_kv_transfer",
    "tests.test_pd_observability",
    "tests.test_pd_process",
    "tests.test_pd_protocol",
    "tests.test_pd_runtime",
    "tests.test_pd_serving",
    "tests.test_pd_shared_slots",
    "tests.test_qwen3_aux_hidden",
    "tests.test_request_abort",
    "tests.test_request_failure",
    "tests.test_scheduler_speculative",
    "tests.test_service_api",
    "tests.test_service_factory",
    "tests.test_service_runtime",
    "tests.test_speculative_sampling",
    "tests.test_target_verify_graph",
    "tests.test_test_tiers",
)

EXTENDED_ONLY_MODULES = (
    "tests.test_benchmark_core",
    "tests.test_benchmark_environment",
    "tests.test_benchmark_run_suite",
    "tests.test_benchmark_runtime",
    "tests.test_benchmark_serve",
    "tests.test_benchmark_suite",
    "tests.test_moe_reference_parity",
    "tests.test_moe_backend_benchmark",
    "tests.test_moe_gptq_gemm_benchmark",
    "tests.test_moe_gate_up_summary",
    "tests.test_moe_profile",
    "tests.test_optional_dependencies",
    "tests.test_pd_batch_example",
    "tests.test_service_overload_benchmark",
    "tests.test_service_startup",
    "tests.test_stage_profiler",
)

GPU_MODULES = (
    "tests.test_eagle3_target_bridge",
    "tests.test_eagle3_verify",
    "tests.test_model_runner_speculative",
    "tests.test_moe_gptq",
    "tests.test_qwen3_aux_hidden",
    "tests.test_target_verify_graph",
)


def modules_for_tier(tier: str) -> tuple[str, ...]:
    if tier == "core":
        return CORE_MODULES
    if tier == "extended":
        return CORE_MODULES + EXTENDED_ONLY_MODULES
    if tier == "gpu":
        return GPU_MODULES
    raise ValueError(f"unknown test tier: {tier}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tier", choices=("core", "extended", "gpu"))
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    modules = modules_for_tier(args.tier)
    if args.list:
        print("\n".join(modules))
        return 0

    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
