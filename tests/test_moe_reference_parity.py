import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch


class MoeReferenceParityTest(unittest.TestCase):

    def test_candidate_worker_forwards_marlin_provider_to_llmserve(self):
        from benchmarks.moe_reference import _run_llmserve_worker

        calls = {}

        class FakeEngine:
            def __init__(self, model, **kwargs):
                calls["model"] = model
                calls["kwargs"] = kwargs
                self.model_runner = type("Runner", (), {})()

            def generate(self, prompts, sampling_params, use_tqdm):
                del prompts, sampling_params, use_tqdm
                return [{"token_ids": [101, 102]}]

            def exit(self):
                calls["exited"] = True

        payload = {
            "model": "/models/qwen3-moe-gptq",
            "cases": [{"id": "case", "prompt_token_ids": [1, 2]}],
            "max_new_tokens": 2,
            "max_model_len": 256,
            "max_num_batched_tokens": 128,
            "gpu_memory_utilization": 0.85,
            "seed": 7,
            "candidate_gptq_backend": "marlin",
            "candidate_marlin_library": "/opt/vllm/_C.abi3.so",
            "candidate_enable_gate_up_fusion": True,
        }

        with patch("llmserve.LLM", FakeEngine):
            result = _run_llmserve_worker(payload)

        self.assertEqual(calls["model"], payload["model"])
        self.assertEqual(calls["kwargs"]["gptq_backend"], "marlin")
        self.assertEqual(
            calls["kwargs"]["marlin_library"],
            "/opt/vllm/_C.abi3.so",
        )
        self.assertTrue(calls["kwargs"]["enable_moe_gate_up_fusion"])
        self.assertTrue(calls["exited"])
        self.assertEqual(result["cases"][0]["generated_token_ids"], [101, 102])

    def test_load_prompt_cases_preserves_fixture_order_and_token_ids(self):
        from benchmarks.moe_reference import load_prompt_cases

        class FakeTokenizer:
            def encode(self, prompt, add_special_tokens):
                if add_special_tokens:
                    raise AssertionError("prompt fixtures must not add special tokens")
                return {"first": [11, 12], "second": [21]}[prompt]

        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "prompts.json"
            fixture.write_text(
                json.dumps(
                    [
                        {"id": "first-case", "prompt": "first"},
                        {"id": "second-case", "prompt": "second"},
                    ]
                ),
                encoding="utf-8",
            )

            cases = load_prompt_cases(
                tokenizer=FakeTokenizer(),
                prompt=None,
                prompts_file=fixture,
            )

        self.assertEqual(
            cases,
            [
                {"id": "first-case", "prompt_token_ids": [11, 12]},
                {"id": "second-case", "prompt_token_ids": [21]},
            ],
        )

    def test_vllm_worker_command_isolated_to_its_reference_directory(self):
        from benchmarks.moe_reference import build_worker_command

        command, environment = build_worker_command(
            worker="vllm",
            payload_path=Path("/tmp/payload.json"),
            output_path=Path("/tmp/vllm.json"),
            reference_package_dir=Path("/opt/reference-vllm"),
            inherited_environment={"PYTHONPATH": "/existing/path"},
        )

        self.assertEqual(command[-2:], ["--worker", "vllm"])
        self.assertIn("/tmp/payload.json", command)
        self.assertIn("/tmp/vllm.json", command)
        self.assertEqual(
            environment["PYTHONPATH"],
            "/opt/reference-vllm:/existing/path",
        )

    def test_compares_matching_greedy_token_traces(self):
        from benchmarks.moe_reference import compare_token_traces

        report = compare_token_traces(
            prompt_token_ids=[151644, 8948, 198],
            reference_token_ids=[100, 101, 102],
            candidate_token_ids=[100, 101, 102],
        )

        self.assertTrue(report["matches"])
        self.assertEqual(report["compared_token_count"], 3)
        self.assertIsNone(report["first_mismatch_index"])

    def test_reports_the_first_greedy_token_mismatch(self):
        from benchmarks.moe_reference import compare_token_traces

        report = compare_token_traces(
            prompt_token_ids=[151644, 8948, 198],
            reference_token_ids=[100, 101, 102],
            candidate_token_ids=[100, 999, 102],
        )

        self.assertFalse(report["matches"])
        self.assertEqual(report["compared_token_count"], 3)
        self.assertEqual(report["first_mismatch_index"], 1)
        self.assertEqual(report["reference_token_id"], 101)
        self.assertEqual(report["candidate_token_id"], 999)

    def test_reports_the_first_mismatched_case_in_a_fixture(self):
        from benchmarks.moe_reference import compare_case_traces

        report = compare_case_traces(
            reference_cases=[
                {
                    "id": "first",
                    "prompt_token_ids": [1],
                    "generated_token_ids": [10, 11],
                },
                {
                    "id": "second",
                    "prompt_token_ids": [2],
                    "generated_token_ids": [20, 21],
                },
            ],
            candidate_cases=[
                {
                    "id": "first",
                    "prompt_token_ids": [1],
                    "generated_token_ids": [10, 11],
                },
                {
                    "id": "second",
                    "prompt_token_ids": [2],
                    "generated_token_ids": [20, 99],
                },
            ],
        )

        self.assertFalse(report["matches"])
        self.assertEqual(report["matched_case_count"], 1)
        self.assertEqual(report["first_mismatch_case_id"], "second")
        self.assertEqual(report["cases"][1]["first_mismatch_index"], 1)

    def test_rejects_traces_with_different_prompt_ids(self):
        from benchmarks.moe_reference import compare_token_traces

        with self.assertRaisesRegex(ValueError, "prompt token ids"):
            compare_token_traces(
                prompt_token_ids=[1, 2],
                reference_token_ids=[3],
                candidate_token_ids=[3],
                candidate_prompt_token_ids=[1, 9],
            )


if __name__ == "__main__":
    unittest.main()
